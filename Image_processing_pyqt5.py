"""
2021/12/31 Properties: "size/page_sel" were added to pdf2image. "add_image" launched.
2021/01/04 "Qsettings" synced. "imshow" added (image2image). "INPUT/OUTPUT" radio button added (image2image).
2021/01/08 "q_jpg" added (image2image). saved_info added (image_processing_GUI.py).
2022/09/19 Load ini file.
2023/01/28 Tuple error fixed.
2023/06/14 Add b_out, g_out, r_out, alpha_out and Debugger
2023/06/14 Add b_out, g_out, r_out, alpha_out and Debugger. v01.01.00
2023/06/24 Add BGR spin box for add_image_tif (b2,g2,r2,b3,g3,r3). v01.01.01

"""

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
from Image_processing_functions import *


# INITIAL SETTINGS #
# True: Load QSettings info.
saved_info = True
ver = 'v01.01.01'


class MainWindow(qtw.QWidget):
    # Store settings
    # settings = qtc.QSettings('my_python', 'Image_processing_pyqt5')
    settings = qtc.QSettings('./Image_processing_pyqt5.ini', qtc.QSettings.IniFormat)
    print('settings file : \n', settings.fileName())
    # Clear settings
    if not saved_info:
        settings.clear()

    # SETTINGS #
    # ---- #
    # tab0 #
    path0 = settings.value('path0')
    dpi = settings.value('dpi', type=int)
    format = settings.value('format', type=str)
    filename = settings.value('filename', type=str)
    filename0 = settings.value('filename0', type=str)
    filename1 = settings.value('filename1', type=str)
    b2, g2, r2 = settings.value('b2', type=int), settings.value('g2', type=int), settings.value('r3', type=int)
    b3, g3, r3 = settings.value('b3', type=int), settings.value('g3', type=int), settings.value('r3', type=int)
    grayscale = settings.value('grayscale', False, type=bool)
    page_off = settings.value('page_off', False, type=bool)
    page_sel0 = settings.value('page_sel0', False, type=bool)
    page_sel1 = settings.value('page_sel1', False, type=bool)
    size_off = settings.value('size_off', False, type=bool)
    size0 = settings.value('size0', type=int)
    size1 = settings.value('size1', type=int)
    thread = settings.value('thread', type=int)
    transparent_pdf = settings.value('transparent_pdf', False, type=bool)
    # ---- #
    # tab1 #
    path1 = settings.value('path1', type=str)
    in_tif = settings.value('in_tif', False, type=bool)
    in_png = settings.value('in_png', False, type=bool)
    in_jpg = settings.value('in_jpg', False, type=bool)
    in_bmp = settings.value('in_bmp', False, type=bool)
    out_tif = settings.value('out_tif', False, type=bool)
    out_png = settings.value('out_png', False, type=bool)
    out_jpg = settings.value('out_jpg', False, type=bool)
    out_bmp = settings.value('out_bmp', False, type=bool)
    point_dic, range_dic = {}, {}
    point_exe0 = settings.value('point_exe0', False, type=bool)
    b0, g0, r0 = settings.value('b0', type=int), settings.value('g0', type=int), settings.value('r0', type=int)
    point_exe1 = settings.value('point_exe1', False, type=bool)
    b1, g1, r1 = settings.value('b1', type=int), settings.value('g1', type=int), settings.value('r1', type=int)
    range_exe0 = settings.value('range_exe0', False, type=bool)
    b_lower, g_lower, r_lower = settings.value('b_lower', type=int), settings.value('g_lower', type=int), settings.value('r_lower', type=int)
    b_upper, g_upper, r_upper = settings.value('b_upper', type=int), settings.value('g_upper', type=int), settings.value('r_upper', type=int)
    b_out, g_out, r_out, alpha_out = settings.value('b_out', type=int), settings.value('g_out', type=int), settings.value('r_out', type=int), settings.value('alpha_out', type=int)
    imshow = settings.value('imshow', False, type=bool)
    q_jpg = settings.value('q_jpg', type=int)


    def __init__(self):
        """MainWindow constructor"""
        super().__init__()

        # Configure the window -------------------------------------------------------
        self.setWindowTitle('IMAGE PROCESSING APP ' + ver)
        # self.resize(600, 600)

        # Create widgets -------------------------------------------------------------
        # ---- #
        # tab0 #
        self.path0_ent = qtw.QLineEdit(self.path0, self, maxLength=99, placeholderText='Enter file path...')
        self.dpi_spn = qtw.QSpinBox(self, maximum=400, minimum=100, singleStep=50)
        self.format_cmb = qtw.QComboBox(self)
        self.filename_ent = qtw.QLineEdit(self.filename, self, placeholderText='Enter filename...')
        self.filename0_ent = qtw.QLineEdit(self.filename0, self, placeholderText='Enter filename...')
        self.filename1_ent = qtw.QLineEdit(self.filename1, self, placeholderText='Enter filename...')
        self.b2_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g2_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r2_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.b3_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g3_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r3_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.grayscale_chk = qtw.QCheckBox('grayscale', self)
        self.page_off_chk = qtw.QCheckBox('page_off', self)
        self.page_sel0_spn = qtw.QSpinBox(self, value=self.page_sel0)
        self.page_sel1_spn = qtw.QSpinBox(self, value=self.page_sel1)
        self.size_off_chk = qtw.QCheckBox('size_off', self)
        self.size0_spn = qtw.QSpinBox(self, value=self.size0, maximum=9999, minimum=9, singleStep=10)
        self.size1_spn = qtw.QSpinBox(self, value=self.size1, maximum=9999, minimum=9, singleStep=10)
        self.thread_spn = qtw.QSpinBox(self, value=self.thread, maximum=8, minimum=1)
        self.pdf2image_btn = qtw.QPushButton('pdf2image', clicked=self.pdf2image_exe)
        self.pdf2image_dir_btn = qtw.QPushButton('pdf2image_dir', clicked=self.pdf2image_dir_exe)
        self.add_image_tif_btn = qtw.QPushButton('add_image_tif', clicked=self.add_image_tif_exe)
        self.add_BefAft_btn = qtw.QPushButton('add_BefAft', clicked=self.add_BefAft_exe)
        # ---- #
        # tab1 #
        self.path1_ent = qtw.QLineEdit(self.path1, self, maxLength=99, placeholderText='Enter file path...')
        self.image2image_btn = qtw.QPushButton('image2image', clicked=self.image2image_exe)
        self.in_tif_rad = qtw.QRadioButton('.tif')
        self.in_png_rad = qtw.QRadioButton('.png')
        self.in_jpg_rad = qtw.QRadioButton('.jpg')
        self.in_bmp_rad = qtw.QRadioButton('.bmp')
        self.out_tif_rad = qtw.QRadioButton('.tif')
        self.out_png_rad = qtw.QRadioButton('.png')
        self.out_jpg_rad = qtw.QRadioButton('.jpg')
        self.out_bmp_rad = qtw.QRadioButton('.bmp')
        self.point_exe0_chk = qtw.QCheckBox('Execute', self)
        self.b0_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g0_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r0_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.point_exe1_chk = qtw.QCheckBox('Execute', self)
        self.b1_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g1_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r1_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.range_exe0_chk = qtw.QCheckBox('Execute', self)
        self.b_lower_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g_lower_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r_lower_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.b_upper_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g_upper_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r_upper_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.b_out_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.g_out_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.r_out_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.alpha_out_spn = qtw.QSpinBox(self, minimum=0, maximum=255)
        self.imshow_chk = qtw.QCheckBox('imshow', self)
        self.q_jpg_spn = qtw.QSpinBox(self, value=100, minimum=1, maximum=100)

        # Configure widgets -------------------------------------------------------------
        # ---- #
        # tab0 #
        # Add event categories
        self.format_cmb.addItems(['.tif', '.jpg', '.png', '.bmp'])
        # ---- #
        # tab1 #

        # Arrange the widgets -----------------------------------------------------------
        # Create main_layout
        main_layout = qtw.QHBoxLayout()
        self.setLayout(main_layout)

        self.tabs = qtw.QTabWidget()
        self.tab0 = qtw.QWidget()
        self.tab1 = qtw.QWidget()

        self.tabs.addTab(self.tab0, 'pdf2image')
        self.tabs.addTab(self.tab1, 'image2image')

        # Tab layout ---------------------------------------------------------------------
        # ---- #
        # tab0 #
        self.tab0.layout = qtw.QVBoxLayout()
        self.tab0.layout.addWidget(qtw.QLabel('<b>Image_processing_pyqt5</b><br>'
                                              ' |- Image_processing_pyqt5.py<br>'
                                              ' |- Image_processing_functions.py'
                                              )
                                   )
        # Execute box #
        execute_form = qtw.QGroupBox('Execute')
        self.tab0.layout.addWidget(execute_form)
        # execute_form_layout
        execute_form_layout = qtw.QGridLayout()
        execute_form_layout.addWidget(qtw.QLabel('# execute_form', self), 1, 1, 1, 10)
        execute_form_layout.addWidget(self.pdf2image_btn, 2, 1, 1, 1)
        execute_form_layout.addWidget(qtw.QLabel(
            ': pdf -> image (jpg/png/tif). Be careful with "grayscale" checkbox.', self), 2, 2, 1, 1)
        execute_form_layout.addWidget(self.pdf2image_dir_btn, 3, 1, 1, 1)
        execute_form_layout.addWidget(qtw.QLabel(
            ': Convert all the pdf files to images in the directory (jpg/png/tif).', self), 3, 2, 1, 1)
        execute_form_layout.addWidget(self.add_image_tif_btn, 4, 1, 1, 1)
        execute_form_layout.addWidget(qtw.QLabel(
            ': Add filename1 on filename0 (tif -> tif).', self), 4, 2, 1, 1)
        execute_form_layout.addWidget(self.add_BefAft_btn, 5, 1, 1, 1)
        execute_form_layout.addWidget(qtw.QLabel(
            ': Add filename1 on filename0 (tif -> tif). Previous: Cyan. Changed: Magenta', self), 5, 2, 1, 1)
        # Set GridLayout to execute_form_layout
        execute_form.setLayout(execute_form_layout)

        # Settings box #
        settings_form = qtw.QGroupBox('Settings')
        self.tab0.layout.addWidget(settings_form)
        # settings_form_layout
        settings_form_layout = qtw.QGridLayout()
        settings_form_layout.addWidget(qtw.QLabel('# settings_form', self), 1, 1, 1, 10)
        settings_form_layout.addWidget(self.path0_ent, 2, 1, 1, 5)  # (row, column, row span, column span)
        settings_form_layout.addWidget(qtw.QLabel('<b># path0</b>', self), 2, 6, 1, 1)
        settings_form_layout.addWidget(self.dpi_spn, 2, 7, 1, 1)
        settings_form_layout.addWidget(qtw.QLabel('<b># dpi</b>', self), 2, 8, 1, 1)
        settings_form_layout.addWidget(self.format_cmb, 2, 9, 1, 1)
        settings_form_layout.addWidget(qtw.QLabel('<b># format</b>', self), 2, 10, 1, 1)
        settings_form_layout.addWidget(self.filename_ent, 5, 1, 1, 3)
        settings_form_layout.addWidget(qtw.QLabel('<b># filename</b> :filename used for pdf2image', self), 5, 4, 1, 4)
        settings_form_layout.addWidget(self.filename0_ent, 6, 1, 1, 3)
        settings_form_layout.addWidget(qtw.QLabel('<b># filename0</b>', self), 6, 4, 1, 2)
        settings_form_layout.addWidget(self.filename1_ent, 6, 6, 1, 3)
        settings_form_layout.addWidget(qtw.QLabel('<b># filename1</b>', self), 6, 9, 1, 2)
        settings_form_layout.addWidget(self.b2_spn, 8, 1, 1, 1)
        settings_form_layout.addWidget(self.g2_spn, 8, 2, 1, 1)
        settings_form_layout.addWidget(self.r2_spn, 8, 3, 1, 1)
        settings_form_layout.addWidget(qtw.QLabel('BGR', self), 8, 4, 1, 1)
        settings_form_layout.addWidget(qtw.QLabel(' --->', self), 8, 5, 1, 1)
        settings_form_layout.addWidget(self.b3_spn, 8, 6, 1, 1)
        settings_form_layout.addWidget(self.g3_spn, 8, 7, 1, 1)
        settings_form_layout.addWidget(self.r3_spn, 8, 8, 1, 1)
        settings_form_layout.addWidget(qtw.QLabel('BGR', self), 8, 9, 1, 1)

        # Set GridLayout to settings_form_layout
        settings_form.setLayout(settings_form_layout)

        # Optional box #
        optional_form = qtw.QGroupBox('Optional')
        self.tab0.layout.addWidget(optional_form)

        # optional_form_layout
        optional_form_layout = qtw.QGridLayout()
        optional_form_layout.addWidget(qtw.QLabel('# optional_form', self), 1, 1, 1, 10)
        optional_form_layout.addWidget(self.grayscale_chk, 2, 1, 1, 1)
        optional_form_layout.addWidget(self.thread_spn, 2, 3, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('<b># thread_count</b>(pdf2image)', self), 2, 4, 1, 1)
        optional_form_layout.addWidget(self.page_off_chk, 3, 1, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('(pdf2image)', self), 3, 2, 1, 1)
        optional_form_layout.addWidget(self.size_off_chk, 3, 6, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('(pdf2image)', self), 3, 7, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('from', self, alignment=qtc.Qt.AlignRight), 4, 2, 1, 1)
        optional_form_layout.addWidget(self.page_sel0_spn, 4, 3, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('<b># page_sel</b> :Don\'t select the same page', self), 4, 4, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('width', self, alignment=qtc.Qt.AlignRight), 4, 7, 1, 1)
        optional_form_layout.addWidget(self.size0_spn, 4, 8, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('to', self, alignment=qtc.Qt.AlignRight), 5, 2, 1, 1)
        optional_form_layout.addWidget(self.page_sel1_spn, 5, 3, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('<b># page_sel</b> :Don\'t select the same page', self), 5, 4, 1, 1)
        optional_form_layout.addWidget(qtw.QLabel('height', self, alignment=qtc.Qt.AlignRight), 5, 7, 1, 1)
        optional_form_layout.addWidget(self.size1_spn, 5, 8, 1, 1)
        # Set GridLayout to optional_form_layout
        optional_form.setLayout(optional_form_layout)
        # Set tab0.layout to tab0
        self.tab0.setLayout(self.tab0.layout)

        # ---- #
        # tab1 #
        self.tab1.layout = qtw.QVBoxLayout()
        self.tab1.layout.addWidget(qtw.QLabel('<b>Image_processing_pyqt5</b><br>'
                                              ' |- Image_processing_pyqt5.py<br>'
                                              ' |- Image_processing_functions.py'
                                              )
                                   )
        # Set tab1.layout
        self.tab1.setLayout(self.tab1.layout)

        # image2image_layout #
        image2image_layout = qtw.QGridLayout()
        # qtw.addWidget(self, row, column, row span, column span)
        image2image_layout.addWidget(qtw.QLabel('# image2image_layout', self), 1, 1, 1, 10)
        image2image_layout.addWidget(self.path1_ent, 2, 1, 1, 9)  # (row, column, row span, column span)
        image2image_layout.addWidget(qtw.QLabel('<b># path1</b>', self), 2, 10, 1, 1)
        image2image_layout.addWidget(self.image2image_btn, 3, 1, 1, 1)
        image2image_layout.addWidget(qtw.QLabel('# Convert all the image in the directory to arbitrary format'),
                                     3, 2, 1, 8)
        image2image_layout.addWidget(self.imshow_chk, 4, 1, 1, 1)
        image2image_layout.addWidget(self.q_jpg_spn, 4, 2, 1, 1)
        image2image_layout.addWidget(qtw.QLabel('<b># q_jpg</b> :Property for IMWRITE_JPEG_QUALITY. int: 1 - 100.', self), 4, 3, 1, 1)
        # Set image2image_layout to tab1.layout
        self.tab1.layout.addLayout(image2image_layout)

        # Input format #
        input_form = qtw.QGroupBox('Input')
        self.tab1.layout.addWidget(input_form)
        # input_form_layout
        input_form_layout = qtw.QGridLayout()
        # qtw.addWidget(self, row, column, row span, column span)
        input_form_layout.addWidget(self.in_tif_rad, 1, 1, 1, 1)
        input_form_layout.addWidget(self.in_png_rad, 1, 2, 1, 1)
        input_form_layout.addWidget(self.in_jpg_rad, 1, 3, 1, 1)
        input_form_layout.addWidget(self.in_bmp_rad, 1, 4, 1, 1)
        # Set input_form_layout to tab1.layout
        input_form.setLayout(input_form_layout)

        # Output format #
        output_form = qtw.QGroupBox('Output')
        self.tab1.layout.addWidget(output_form)
        # output_form_layout
        output_form_layout = qtw.QGridLayout()
        output_form_layout.addWidget(self.out_tif_rad, 1, 1, 1, 1)
        output_form_layout.addWidget(self.out_png_rad, 1, 2, 1, 1)
        output_form_layout.addWidget(self.out_jpg_rad, 1, 3, 1, 1)
        output_form_layout.addWidget(self.out_bmp_rad, 1, 4, 1, 1)
        # Set output_Form_layout to tab1.layout
        output_form.setLayout(output_form_layout)

        # point_form box #
        point_form = qtw.QGroupBox('Point Conversion')
        self.tab1.layout.addWidget(point_form)
        # point_form_layout
        point_form_layout = qtw.QGridLayout()
        # qtw.addWidget(self, row, column, row span, column span)
        # point_exe0
        point_form_layout.addWidget(qtw.QLabel('# point_form\nMake selected color transparent. \nOnly works for output: .png/.tif (#Doesn\'t work...)', self),
                                     1, 1, 1, 10)
        point_form_layout.addWidget(self.point_exe0_chk, 2, 1, 1, 1)
        point_form_layout.addWidget(self.b0_spn, 2, 2, 1, 1)
        point_form_layout.addWidget(qtw.QLabel('B', self), 2, 3, 1, 1)
        point_form_layout.addWidget(self.g0_spn, 2, 4, 1, 1)
        point_form_layout.addWidget(qtw.QLabel('G', self), 2, 5, 1, 1)
        point_form_layout.addWidget(self.r0_spn, 2, 6, 1, 1)
        point_form_layout.addWidget(qtw.QLabel('R', self), 2, 7, 1, 1)
        # point_exe1
        point_form_layout.addWidget(self.point_exe1_chk, 3, 1, 1, 1)
        point_form_layout.addWidget(self.b1_spn, 3, 2, 1, 1)
        point_form_layout.addWidget(qtw.QLabel('B', self), 3, 3, 1, 1)
        point_form_layout.addWidget(self.g1_spn, 3, 4, 1, 1)
        point_form_layout.addWidget(qtw.QLabel('G', self), 3, 5, 1, 1)
        point_form_layout.addWidget(self.r1_spn, 3, 6, 1, 1)
        point_form_layout.addWidget(qtw.QLabel('R', self), 3, 7, 1, 1)
        # Set point_form_layout to point_form
        point_form.setLayout(point_form_layout)

        # range_form box #
        range_form = qtw.QGroupBox('Range Conversion')
        self.tab1.layout.addWidget(range_form)
        # point_form_layout
        range_form_layout = qtw.QGridLayout()
        # qtw.addWidget(self, row, column, row span, column span)
        # range_exe0
        range_form_layout.addWidget(qtw.QLabel('# Make selected color channel to arbitrary color. Only works for output: .png/.tif.', self),
                                       1, 1, 1, 10)
        range_form_layout.addWidget(self.range_exe0_chk, 2, 1, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('<lower>', self, alignment=qtc.Qt.AlignRight), 2, 2, 1, 1)
        range_form_layout.addWidget(self.b_lower_spn, 2, 3, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('B', self), 2, 4, 1, 1)
        range_form_layout.addWidget(self.g_lower_spn, 2, 5, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('G', self), 2, 6, 1, 1)
        range_form_layout.addWidget(self.r_lower_spn, 2, 7, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('R', self), 2, 8, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('<upper>', self, alignment=qtc.Qt.AlignRight), 3, 2, 1, 1)
        range_form_layout.addWidget(self.b_upper_spn, 3, 3, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('B', self), 3, 4, 1, 1)
        range_form_layout.addWidget(self.g_upper_spn, 3, 5, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('G', self), 3, 6, 1, 1)
        range_form_layout.addWidget(self.r_upper_spn, 3, 7, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('R', self), 3, 8, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('<out> ', self, alignment=qtc.Qt.AlignRight), 4, 2, 1, 1)
        range_form_layout.addWidget(self.b_out_spn, 4, 3, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('B', self), 4, 4, 1, 1)
        range_form_layout.addWidget(self.g_out_spn, 4, 5, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('G', self), 4, 6, 1, 1)
        range_form_layout.addWidget(self.r_out_spn, 4, 7, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('R', self), 4, 8, 1, 1)
        range_form_layout.addWidget(self.alpha_out_spn, 4, 9, 1, 1)
        range_form_layout.addWidget(qtw.QLabel('alpha', self), 4, 10, 1, 1)

        # Set range_form_layout to range_form
        range_form.setLayout(range_form_layout)

        # Connect Events --------------------------------------------------------------
        # Sync to Checkbox
        # ---- #
        # tab0 #
        self.page_off_chk.toggled.connect(self.page_sel0_spn.setDisabled)
        self.page_off_chk.toggled.connect(self.page_sel1_spn.setDisabled)
        self.size_off_chk.toggled.connect(self.size0_spn.setDisabled)
        self.size_off_chk.toggled.connect(self.size1_spn.setDisabled)
        self.page_off_chk.setChecked(True)
        self.size_off_chk.setChecked(True)
        # Sync to Qsettings
        if saved_info == True:
            # ---- #
            # tab0 #
            self.dpi_spn.setValue(self.dpi)
            self.format_cmb.setCurrentText(self.format)
            self.b2_spn.setValue(self.b2)
            self.g2_spn.setValue(self.g2)
            self.r2_spn.setValue(self.r2)
            self.b3_spn.setValue(self.b3)
            self.g3_spn.setValue(self.g3)
            self.r3_spn.setValue(self.r3)
            self.grayscale_chk.setChecked(self.grayscale)
            self.page_off_chk.setChecked(self.page_off)
            self.page_sel0_spn.setValue(self.page_sel0)
            self.page_sel1_spn.setValue(self.page_sel1)
            self.size_off_chk.setChecked(self.size_off)
            self.size0_spn.setValue(self.size0)
            self.size1_spn.setValue(self.size1)
            self.thread_spn.setValue(self.thread)
            # ---- #
            # tab1 #
            # Sync to Qsettings
            self.in_tif_rad.setChecked(self.in_tif)
            self.in_png_rad.setChecked(self.in_png)
            self.in_jpg_rad.setChecked(self.in_jpg)
            self.in_bmp_rad.setChecked(self.in_bmp)
            self.out_tif_rad.setChecked(self.out_tif)
            self.out_png_rad.setChecked(self.out_png)
            self.out_jpg_rad.setChecked(self.out_jpg)
            self.out_bmp_rad.setChecked(self.out_bmp)
            self.point_exe0_chk.setChecked(self.point_exe0)
            self.b0_spn.setValue(self.b0)
            self.g0_spn.setValue(self.g0)
            self.r0_spn.setValue(self.r0)
            self.point_exe1_chk.setChecked(self.point_exe1)
            self.b1_spn.setValue(self.b1)
            self.g1_spn.setValue(self.g1)
            self.r1_spn.setValue(self.r1)
            self.range_exe0_chk.setChecked(self.range_exe0)
            self.b_lower_spn.setValue(self.b_lower)
            self.g_lower_spn.setValue(self.g_lower)
            self.r_lower_spn.setValue(self.r_lower)
            self.b_upper_spn.setValue(self.b_upper)
            self.g_upper_spn.setValue(self.g_upper)
            self.r_upper_spn.setValue(self.r_upper)
            self.b_out_spn.setValue(self.b_out)
            self.g_out_spn.setValue(self.g_out)
            self.r_out_spn.setValue(self.r_out)
            self.alpha_out_spn.setValue(self.alpha_out)
            self.imshow_chk.setChecked(self.imshow)
            self.q_jpg_spn.setValue(self.q_jpg)
        # Set tabs to main_layout ------------------------------------------------------
        main_layout.addWidget(self.tabs)
        # End main UI code -------------------------------------------------------------
        self.show()

    # Functions -------------------------------------------------------------------------
    # ---- #
    # tab0 #
    def pdf2image_exe(self):
        page_sel = [self.page_sel0_spn.value(), self.page_sel1_spn.value()]
        # size
        if self.size_off_chk.isChecked():
            size = None
        elif not self.size_off_chk.isChecked():
            size = (self.size0_spn.value(), self.size1_spn.value())
        else:
            Exception('Exception occurred in size')

        pdf2image(
            path=self.path0_ent.text(),
            dpi=self.dpi_spn.value(),
            filename=self.filename_ent.text(),
            format=self.format_cmb.currentText(),
            page_off=self.page_off_chk.isChecked(),
            page_sel=page_sel,
            grayscale=self.grayscale_chk.isChecked(),
            size=size,
            thread_count=self.thread_spn.value(),
        )

    def pdf2image_dir_exe(self):
        pdf2image_dir(
            path=self.path0_ent.text(),
            dpi=self.dpi_spn.text(),
            format=self.format_cmb.currentText(),
            page_off=self.page_off_chk.isChecked(),
            grayscale=self.grayscale_chk.isChecked()
        )

    def add_image_tif_exe(self):
        add_image_tif(
            path=self.path0_ent.text(),
            filename0=self.filename0_ent.text(),
            filename1=self.filename1_ent.text(),
            grayscale=self.grayscale_chk.isChecked(),
            b3=self.b3_spn.value(),
            g3=self.g3_spn.value(),
            r3=self.r3_spn.value(),
        )

    def add_BefAft_exe(self):
        add_BefAft(
            path=self.path0_ent.text(),
            filename0=self.filename0_ent.text(),
            filename1=self.filename1_ent.text(),
            grayscale=self.grayscale_chk.isChecked()
        )

    # ---- #
    # tab1 #
    def image2image_exe(self):
        # Put number in dictionary when Execute checked.
        if self.point_exe0_chk:
            self.point_dic['b0'] = self.b0_spn.value()
            self.point_dic['g0'] = self.g0_spn.value()
            self.point_dic['r0'] = self.g0_spn.value()
        if self.point_exe1_chk:
            self.point_dic['b1'] = self.b1_spn.value()
            self.point_dic['g1'] = self.g1_spn.value()
            self.point_dic['r1'] = self.r1_spn.value()
        if self.range_exe0_chk:
            self.range_dic['b_lower'] = self.b_lower_spn.value()
            self.range_dic['g_lower'] = self.g_lower_spn.value()
            self.range_dic['r_lower'] = self.r_lower_spn.value()
            self.range_dic['b_upper'] = self.b_upper_spn.value()
            self.range_dic['g_upper'] = self.g_upper_spn.value()
            self.range_dic['r_upper'] = self.r_upper_spn.value()
            self.range_dic['b_out'] = self.b_out_spn.value()
            self.range_dic['g_out'] = self.g_out_spn.value()
            self.range_dic['r_out'] = self.r_out_spn.value()
            self.range_dic['alpha_out'] = self.alpha_out_spn.value()
        # INPUT file format
        if self.in_tif_rad.isChecked():
            input = '.tif'
        elif self.in_png_rad.isChecked():
            input = '.png'
        elif self.in_jpg_rad.isChecked():
            input = '.jpg'
        elif self.in_bmp_rad.isChecked():
            input = '.bmp'
        else:
            Exception('"input" unselected!!')
        # OUTPUT file format
        if self.out_tif_rad.isChecked():
            output = '.tif'
        elif self.out_png_rad.isChecked():
            output = '.png'
        elif self.out_jpg_rad.isChecked():
            output = '.jpg'
        elif self.out_bmp_rad.isChecked():
            output = '.bmp'
        else:
            Exception('"output" unselected!!')
        print('Execute image2image_exe.')

        image2image(
            path=self.path1_ent.text(),
            point_exe0=self.point_exe0_chk.isChecked(),
            point_exe1=self.point_exe1_chk.isChecked(),
            range_exe0=self.range_exe0_chk.isChecked(),
            input=input,
            output=output,
            point_dic=self.point_dic,
            range_dic=self.range_dic,
            imshow=self.imshow_chk.isChecked(),
            q_jpg=self.q_jpg_spn.value()
        )


    # ALL #
    def closeEvent(self, e):
        self.saveSettings()

    # Save settings info
    def saveSettings(self):
        # settings = qtc.QSettings('my_python', 'Image_processing_GUI')
        # ---- #
        # tab0 #
        self.settings.setValue('path0', self.path0_ent.text())
        self.settings.setValue('dpi', self.dpi_spn.value())
        self.settings.setValue('format', self.format_cmb.currentText())
        self.settings.setValue('filename', self.filename_ent.text())
        self.settings.setValue('filename0', self.filename0_ent.text())
        self.settings.setValue('filename1', self.filename1_ent.text())
        self.settings.setValue('b2', self.b2_spn.value())
        self.settings.setValue('g2', self.g2_spn.value())
        self.settings.setValue('r2', self.r2_spn.value())
        self.settings.setValue('b3', self.b3_spn.value())
        self.settings.setValue('g3', self.g3_spn.value())
        self.settings.setValue('r3', self.r3_spn.value())
        self.settings.setValue('grayscale', self.grayscale_chk.isChecked())
        self.settings.setValue('page_off', self.page_off_chk.isChecked())
        self.settings.setValue('page_sel0', self.page_sel0_spn.value())
        self.settings.setValue('page_sel1', self.page_sel1_spn.value())
        self.settings.setValue('size_off', self.size_off_chk.isChecked())
        self.settings.setValue('size0', self.size0_spn.value())
        self.settings.setValue('size1', self.size1_spn.value())
        self.settings.setValue('thread', self.thread_spn.value())
        # ---- #
        # tab1 #
        self.settings.setValue('path1', self.path1_ent.text())
        self.settings.setValue('in_tif', self.in_tif_rad.isChecked())
        self.settings.setValue('in_png', self.in_png_rad.isChecked())
        self.settings.setValue('in_jpg', self.in_jpg_rad.isChecked())
        self.settings.setValue('in_bmp', self.in_bmp_rad.isChecked())
        self.settings.setValue('out_tif', self.out_tif_rad.isChecked())
        self.settings.setValue('out_png', self.out_png_rad.isChecked())
        self.settings.setValue('out_jpg', self.out_jpg_rad.isChecked())
        self.settings.setValue('out_bmp', self.out_bmp_rad.isChecked())
        self.settings.setValue('point_exe0', self.point_exe0_chk.isChecked())
        self.settings.setValue('b0', self.b0_spn.value())
        self.settings.setValue('g0', self.g0_spn.value())
        self.settings.setValue('r0', self.r0_spn.value())
        self.settings.setValue('point_exe1', self.point_exe1_chk.isChecked())
        self.settings.setValue('b1', self.b1_spn.value())
        self.settings.setValue('g1', self.g1_spn.value())
        self.settings.setValue('r1', self.r1_spn.value())
        self.settings.setValue('range_exe0', self.range_exe0_chk.isChecked())
        self.settings.setValue('b_lower', self.b_lower_spn.value())
        self.settings.setValue('g_lower', self.g_lower_spn.value())
        self.settings.setValue('r_lower', self.r_lower_spn.value())
        self.settings.setValue('b_upper', self.b_upper_spn.value())
        self.settings.setValue('g_upper', self.g_upper_spn.value())
        self.settings.setValue('r_upper', self.r_upper_spn.value())
        self.settings.setValue('b_out', self.b_out_spn.value())
        self.settings.setValue('g_out', self.g_out_spn.value())
        self.settings.setValue('r_out', self.r_out_spn.value())
        self.settings.setValue('alpha_out', self.alpha_out_spn.value())
        self.settings.setValue('imshow', self.imshow_chk.isChecked())
        self.settings.setValue('q_jpg', self.q_jpg_spn.value())


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())





