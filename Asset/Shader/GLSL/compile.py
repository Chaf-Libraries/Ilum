import os
import subprocess

def main():
    file_list=os.listdir()

    target_shader=[]

    for file in file_list:
        ext = file.split('.')[-1]
        if ext == 'vert' or ext == 'frag' or ext == 'comp' or ext == 'tesc' or ext == 'tese':
            target_shader.append(file)
    
    for shader in target_shader:
        result = os.popen("glslc " + shader +" -o " + shader + ".spv").read()
        if result == "":
            print(shader + " compile successfully!")
        else:
            print(result)
    print("Compile complete!")
    input()

if __name__ == '__main__':
    main()