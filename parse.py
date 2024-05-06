import pprint
from typing import List

import docx
from docx.document import Document
from docx.section import Section
from docx.table import Table
from docx.text.paragraph import Paragraph

# remove blank lines as following will break section structure
# def remove_blank_lines_in_section(doc: Document, section_index: int):
#     """
#     Removes all blank lines within the specified section of the Word document.
#     """
#     section = doc.sections[section_index]

#     # Iterate over paragraphs in the section
#     for paragraph in section.iter_inner_content():
#     # for paragraph in list(section.iter_inner_content())[:-1]:
#         if not paragraph.text.strip():
#             # Remove the blank paragraph
#             # paragraph._element.getparent().remove(paragraph._element)
#             paragraph._element = None

# def remove_empty_paragraphs(doc: Document):
#     """
#     Remove all empty paragraphs from the given Word document.
#     """
#     for paragraph in doc.paragraphs[::-1]:  # Iterate over paragraphs in reverse order
#         if not paragraph.text.strip():  # Check if the paragraph is empty
#             paragraph._element.getparent().remove(
#                 paragraph._element
#             )  # Remove the paragraph element

def show_doc(doc: Document, sec_range=(None, None), show_empty=False):
    for i, sec in enumerate(doc.sections[sec_range[0]:sec_range[1]]):
        print(f"----- Section {i} -----")
        show_sec(sec, show_empty)

def show_sec(sec: Section, show_empty=False):
    for i, para in enumerate(sec.iter_inner_content()):
        show_para(para, show_empty)

def show_para(para: Paragraph|Table, show_empty=False):
    if isinstance(para, Table):
        print(para.table)
    else:
        left_indent = para.paragraph_format.left_indent
        first_line_indent = para.paragraph_format.first_line_indent
        left_indent = left_indent.pt if left_indent else 0
        first_line_indent = first_line_indent.pt if first_line_indent else 0
        if not show_empty:
            if para.text.strip():
                print(left_indent, first_line_indent, para.text)
        else:
            print(left_indent, first_line_indent, para.text)

def para_type(para: Paragraph):
    left_indent = para.paragraph_format.left_indent
    first_line_indent = para.paragraph_format.first_line_indent
    left_indent = left_indent.pt if left_indent else 0
    first_line_indent = first_line_indent.pt if first_line_indent else 0
    if left_indent:
        return "小节标题"
    if first_line_indent:
        return "正文开篇"
    return "正文续篇"
