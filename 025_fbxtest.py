import pyfbx

# Load the FBX file
filename = "your_file_with_skeleton.fbx"
scene = pyfbx.load(filename)

# Traverse nodes to find skeletons
def print_skeleton_info(node, depth=0):
    if node.type == 'Skeleton':
        print("  " * depth + f"Skeleton Node: {node.name}")

    for child in node.children:
        print_skeleton_info(child, depth + 1)

# Start with the root node
root_node = scene.root
print_skeleton_info(root_node)

# Don't forget to release resources
pyfbx.release(scene)