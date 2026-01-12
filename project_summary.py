import os

def format_size(size):
    for unit in ['B','KB','MB','GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"

project_dir = "."

output = []

for root, dirs, files in os.walk(project_dir):
    level = root.replace(project_dir, "").count(os.sep)
    indent = " " * 4 * level
    output.append(f"{indent}{os.path.basename(root)}/")

    subindent = " " * 4 * (level + 1)
    for f in files:
        file_path = os.path.join(root, f)
        size = os.path.getsize(file_path)
        output.append(f"{subindent}{f}  ({format_size(size)})")

# Save to file
with open("project_structure.txt", "w") as file:
    file.write("\n".join(output))

print("✅ Project structure saved to project_structure.txt")
