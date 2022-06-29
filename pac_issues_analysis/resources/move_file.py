
# You can use the os module alone to do this.

import os
from shutil import copyfile
from nltk import word_tokenize
from collections import defaultdict
source_dir = "/Users/shiyishen/PycharmProjects/PAC_Ident_DATA_Prep/output.txt"
dest_dir = "/Users/shiyishen/PycharmProjects/PAC_Ident_DATA_Prep/redo"

print('hello')
issue = defaultdict()
for top, dirs, files in  os.walk(source_dir):
  count = 0
  for filename in files:
    count +=1
    file_path = os.path.join(top,filename)
    with open(file_path, 'r',encoding='utf-8',errors='ignore') as f_in:
      text = f_in.read()
      # print(text)
      if text.lower().__contains__('issue'):
        issue[filename] = text
        os.rename(os.path.join(source_dir,filename),os.path.join(source_dir,str(count)))
      else:
        f_in.close()


for name,value in issue.items():
  with open(os.path.join(dest_dir,name),'w+') as f_out:
    f_out.write(value)
    f_out.close()
