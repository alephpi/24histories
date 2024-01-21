def merge():
  with open('ancient.dict', 'r', encoding='utf-8') as f:
    ancient_codes = f.readlines()
    ancient_codes = set([int(code, 16) for code in ancient_codes])
  with open('gbk.dict', 'r', encoding='utf-8') as f:
    gbk_codes = f.readlines()
    gbk_codes = set([int(code, 16) for code in gbk_codes])
  with open('cn.dict', 'w', encoding='utf-8') as f:
    cn_codes = gbk_codes.intersection(ancient_codes)
    cn_chars = [chr(code) for code in cn_codes]
    cn_codes = [hex(code)[2:].zfill(4) for code in cn_codes]
    for code,char in zip(cn_codes,cn_chars):
      f.write(f"{code}\t{char}\n")

def punc():
  PUNCS = '、，。？！：；‘’“”（）【】——·〔〕《》' + '〇' # 对，当初设计unicode的时候错把汉字〇当成标点了
  with open('punc.dict','w',encoding='utf-8') as f:
    for c in PUNCS:
      f.write(f"{ord(c):04X}\t{c}\n")
if __name__ == '__main__':
  merge()
  punc()