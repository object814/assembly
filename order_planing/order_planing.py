import os
from gdino import GroundingDINOAPIWrapper, visualize, my_visualize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from openai import OpenAI
import json


def annotate(img_path, output_path, text_prompt="board.stick", box_threshold=0.2):
    token = os.environ["DINO_API_KEY"]  # 如果没有设置该环境变量，会抛出 KeyError
    gdino = GroundingDINOAPIWrapper(token)

    def bbox_area(bbox):  # bbox: [x0, y0, x1, y1]
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def PIL_area(image_pil):
        return image_pil.size[0] * image_pil.size[1]

    # get results
    prompts = dict(image=img_path, prompt=text_prompt)
    image_pil = Image.open(prompts["image"])
    while True:
        results = gdino.inference(prompts, return_mask=True, box_threshold=box_threshold)
        redo = False  # 如果检测到的物体面积太大就重新做一遍
        for box in results["boxes"]:
            if bbox_area(box) > 0.8 * PIL_area(image_pil):
                redo = True
                break
        if not redo:
            break
        else:
            print(f"[INFO] Redoing gdino inference because the detected object is too large. len(results)= {len(results['boxes'])}")

    # visualize the results
    image_pil = my_visualize(image_pil, results)
    image_pil = image_pil.convert("RGB")
    image_np = np.array(image_pil)
    plt.imsave(output_path, image_np)

    return results


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_message(image_path):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encode_image(image_path)}",
            "detail": "high"
        },
    }


def order_planing(
        img_path = "img/input.png",
        output_path = "img/output.png",
        text_prompt = "board.stick",
        manual_img_path = "img/manual.png",
        prompt_path = "prompts/order_planing.txt"
    ):
    # adjust path
    img_path = os.path.join(os.path.dirname(__file__), img_path)
    output_path = os.path.join(os.path.dirname(__file__), output_path)
    manual_img_path = os.path.join(os.path.dirname(__file__), manual_img_path)
    prompt_path = os.path.join(os.path.dirname(__file__), prompt_path)

    # SoM annotation
    results = annotate(img_path, output_path, text_prompt)

    # GPT generate assembly order
    client = OpenAI()
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    image_message(manual_img_path),  # 说明书图片
                    image_message(output_path),  # 标注过的图片
                ],
            }
        ],
        max_tokens=1000,
        temperature=0,
    )
    final_output = response.choices[0].message.content.replace("```json", "").replace("```", "")
    order_list = json.loads(final_output)
    with open(os.path.join(os.path.dirname(__file__), "img", "order_list.json"), "w") as f:
        json.dump(order_list, f, indent=4)
    order_list = [x-1 for x in order_list]
    mask_list = [results["masks"][i] for i in order_list]  # PIL
    bbox_list = [results["boxes"][i] for i in order_list]  # [x0, y0, x1, y1]
    with open(os.path.join(os.path.dirname(__file__), "img", "bbox_list.json"), "w") as f:
        json.dump(bbox_list, f, indent=4)

    # transfer mask from PIL to binary np array
    mask_list = [np.array(mask) for mask in mask_list]
    alpha_channel_list = [mask[:, :, 3] for mask in mask_list]
    binary_mask_list = [alpha_channel > 0 for alpha_channel in alpha_channel_list]

    # visualize
    image_pil = Image.open(output_path)
    image_pil = visualize(image_pil, results, return_mask=True, draw_score=False, draw_index=False)
    image_pil = image_pil.convert("RGB")
    image_np = np.array(image_pil)
    plt.imsave(os.path.join(os.path.dirname(__file__), "img", "output_mask.png"), image_np)

    return binary_mask_list, bbox_list


if __name__ == "__main__":
    result_list = order_planing()
    print(len(result_list))
    print(result_list[0])
    """
    bbox format: (x0, y0, x1, y1)
    Sample Output:
    [
        [
            192.2847442626953,
            103.83705139160156,
            363.1347961425781,
            262.05517578125
        ],
        [
            99.5417709350586,
            90.45260620117188,
            140.64987182617188,
            259.27374267578125
        ],
        [
            404.93609619140625,
            89.23165893554688,
            449.3974304199219,
            258.9975891113281
        ],
        [
            487.5415344238281,
            78.55289459228516,
            540.9432373046875,
            250.82534790039062
        ],
        [
            2.9393601417541504,
            90.1539077758789,
            53.38908004760742,
            260.6542053222656
        ]
    ]
    """