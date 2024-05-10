from typing import List

from docx.text.paragraph import Paragraph
from parse import para_property


def para_type(para: Paragraph):
    left_indent, first_line_indent, font_size = para_property(para)
    if left_indent:
        return "小节标题"
    if first_line_indent:
        return "正文开篇"
    return "正文续篇"


def process_pages(pages: List[List[List[Paragraph]]]):
    book: List[Chapter] = []
    chapter = Chapter()
    last_section = None
    section = Section()
    for page in pages:
        header = page.pop(0)
        for sec in page:
            for para in sec:
                # if para is empty, create a new section
                text = para.text.strip()
                if not text:
                    chapter.push_section(section)
                    if last_section:
                        last_section.distribute()
                    last_section = section
                    section = Section()
                else:
                    left_indent, first_line_indent, font_size = para_property(para)
                    # 章节标题
                    if left_indent > 0:
                        chapter = Chapter()
                        titles = [para.text.strip() for para in sec]
                        chapter.push_title(titles)
                        book.append(chapter)
                    # 章节正文，开篇
                    elif first_line_indent > font_size:
                        section.paras.append(para)
                    # 小节标题为页眉子串
                    elif text in header:
                        section.title = text
                    # 章节正文，续篇
                    else:
                        last_section.residues.append(text)
    return book

class Section:
    def __init__(self):
        self.title: str = ""
        self.paras: List[str] = []
        self.wenyans: List[str] = []
        self.baihuas: List[str] = []
        self.residues:List[str] = []
        # self.wenyan_len = 0
        # self.baihua_len = 0

    # def push_wenyan(self, paragraph: str):
    #     self.wenyans.append(paragraph)
    #     self.wenyans_len += 1

    # def push_baihua(self, paragraph: str):
    #     self.baihuas.append(paragraph)
    #     self.baihua_len += 1
    
    def distribute(self):
        l = len(self.paras)
        assert l % 2 == 0
        # distribute odd paragraphs to wenyan, even paragraphs to baihua
        self.wenyans = self.paras[:l//2]
        self.baihuas = self.paras[l//2:]
        if len(self.residues) == 1:
            self.baihuas[-1] += self.residues[0]
        elif len(self.residues) == 2:
            self.wenyans[-1] += self.residues[0]
            self.baihuas[-1] += self.residues[1]
        else:
            raise ValueError("residues length should be 1 or 2")


# class Page:
#     def __init__(self, page: List[Paragraph]):
#         self.page = page
#         self.header = page.pop(0)
#         self.sections: List[List[Paragraph]] = []
#         section: List[Paragraph] = []
        

class Chapter:
    def __init__(self, titles):
        self.titles = titles
        self.sections: List[Section] = []
    
    def push_section(self, section: Section):
        self.sections.append(section)