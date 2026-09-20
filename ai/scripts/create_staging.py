import random
import json
from pathlib import Path

# Generate a directory name with a random integer
random.seed(42)
random_int = random.randint(0, 99)
dir_name = f"ops-staging-{random_int}"

# Create the directory
Path(dir_name).mkdir(parents=True, exist_ok=True)
print(f"Created directory: {dir_name}")

# Create .gitkeep file inside the directory
( Path(dir_name) / ".gitkeep" ).write_text("")
print(f"Created .gitkeep in {dir_name}")

# Create package.json with minimal content
package_json_content = {
    "name": "ops-staging",
    "version": "0.1.0",
    "description": "Temporary staging package for Ops.",
    "main": "index.js"
}
(Path(dir_name) / "package.json").write_text(json.dumps(package_json_content, indent=2))
print(f"Created package.json in {dir_name}")

from time import sleep
sleep(1)
print("All files created successfully.\n")
