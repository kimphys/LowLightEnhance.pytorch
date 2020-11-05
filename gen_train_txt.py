import os;

img_dir_list = ['test']

file_list = [];
img_file_list = [];

f = open('test.txt','w')
count = 0
for img_dir in img_dir_list:
    file_list = os.listdir(img_dir);
    img_file_list = [file for file in file_list if file.endswith((".jpg",".png",".tiff",".tif",".bmp"))]

    for i in range(0,len(img_file_list)):
        # if count > 5: 
        #     f.close()
        #     break
        print(os.getcwd() + '/' + img_dir +"/"+img_file_list[i]);
        data = os.getcwd() + '/' + img_dir +"/"+img_file_list[i] +'\n';
        f.write(data)
        count += 1
f.close()
