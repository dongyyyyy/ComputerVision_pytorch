# csv_filepath = '/home/eslab/kdy/Vision_transformer/class_info.csv'

# class_information = dict()
# class_df = pd.read_csv(csv_filepath)
# for v in class_df.values:
#     class_name, class_num = v
#     class_information[class_name] = class_num

# import pandas as pd

# ImageNet_class_information_filepath = '/home/eslab/kdy/Vision_transformer/ImageNet_class_information.txt'

# def check_class_information(txt_path):
#     data = pd.read_csv(ImageNet_class_information_filepath,sep='\t',engine='python',encoding='utf-8')

#     class_information = dict()
#     filename, class_number, class_name = data.columns[0].split(' ')

#     class_information[filename] = [class_number,class_name]

#     for v in data.values:
        
#         filename, class_number, class_name = str(v)[2:-2].split(' ')
#         class_information[filename] = [int(class_number)-1,class_name]
#     return class_information

# # information of ImageNet dataset about class
# class_information = check_class_information(ImageNet_class_information_filepath)

# print(class_information)