import subprocess

def generate_plantuml_image(uml_file_path, output_image_path):
    # 调用PlantUML生成图片
    plantuml_jar_path = "plantuml-1.2024.7.jar"  # 替换为你的PlantUML jar文件路径
    subprocess.run(["java", "-jar", plantuml_jar_path, uml_file_path])
    
    # 移动生成的图片到指定路径
    generated_image_path = uml_file_path.replace(".puml", ".png")
    subprocess.run(["mv", generated_image_path, output_image_path])
