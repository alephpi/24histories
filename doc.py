from typing import List


class Section:
    def __init__(self, sec_title=None):
        self.title = sec_title
        self.wenyans: List[str] = []
        self.baihuas: List[str] = []
        self.wenyan_len = 0
        self.baihua_len = 0

    def push_wenyan(self, paragraph: str):
        self.wenyans.append(paragraph)
        self.wenyans_len += 1

    def push_baihua(self, paragraph: str):
        self.baihuas.append(paragraph)
        self.baihua_len += 1


class Chapter:
    def __init__(self, title1, title2, title3):
        self.title1 = title1
        self.title2 = title2
        self.title3 = title3
        self.sections: List[Section] = []

    def push_section(self, section: Section):
        self.sections.append(section)