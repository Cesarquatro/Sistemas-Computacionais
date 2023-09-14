import os
import numpy as np

#list directory
old_names = os.listdir(path=r'src\felipeli')
#print(old_names)
print(f"Tamanho: {len(old_names)}")

extension = old_names

for i in range(len(old_names)):
    # Split file name and extension
    old_names[i], extension[i] = os.path.splitext('.')
    # Rename file
    print(f'src\\felipeli\\{old_names[i]}')
    os.rename(f'src\\felipeli\\{old_names[i]}', f'src\\Felipeli\\media{i}.{extension[i]}')
    print(f"media{i}.{extension[i]} is saved")

#print html lines
#for i in range(len(old_names)):
#    print(f'<img src=r"src\felipeli\media{i}.{extension[i]}" alt="media{i}.{extension[i]}" width="200" height="200">')