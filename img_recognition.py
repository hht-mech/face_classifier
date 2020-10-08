from PIL import Image
from zipfile import ZipFile
import numpy as np
import cv2 as cv
import pytesseract

face_cascades = cv.CascadeClassifier("readonly/haarcascade_frontalface_default.xml")

def analyze_newspaper(zipfile):
    newspapers = []
    with ZipFile(zipfile) as myzip:
        for file in myzip.infolist():
            page_info = {}
            page_info["filename"] = file.filename
            print("Analyzing {}".format(page_info["filename"]))

            img = Image.open(myzip.open(file))
            page_info["image"] = img

            img = img.convert("L")
            detected_strings = pytesseract.image_to_string(img)
            page_info["text"] = detected_strings

            cv_img = np.array(img.convert("RGB"))
            cv_img = cv_img[:,:,::-1].copy()
            gray = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
            page_info["gray"] = gray
            
            newspapers.append(page_info)
            
    print("Analyzing Finished")
    return newspapers

def search_for_text(text,newspaper_lst, scale_factor):
    for newspaper in newspaper_lst:
        bounding_boxes = face_cascades.detectMultiScale(newspaper['gray'], scale_factor, 5).tolist()
        if text in newspaper['text']:
            print('Results found in file {}'.format(newspaper['filename']))
            if len(bounding_boxes) == 0:
                print('But there were no faces in that file!')
            else:
                rows_size = (len(bounding_boxes) - 1)//5 + 1
                first_image = newspaper['image'].crop((0, 0, 100, 100))
                first_image.thumbnail((100, 100))
                contact_sheet= Image.new(first_image.mode, (first_image.width*5,first_image.height*rows_size))
                x = 0
                y = 0

                for bounding_box in bounding_boxes:
                    num_array = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
                    box = np.dot(bounding_box, num_array)
                    img = newspaper['image'].crop(box)
                    img.thumbnail((100, 100), Image.ANTIALIAS)
                    contact_sheet.paste(img, (x, y))
                    
                    if x+first_image.width == contact_sheet.width:
                        x = 0
                        y += first_image.height
                    else:
                        x += first_image.width

                contact_sheet.show()
    return None

zipfile = "readonly/small_img.zip"
newspapers = analyze_newspaper(zipfile)
search_for_text("Christopher", newspapers, 1.3)

zipfile1 = "readonly/images.zip"
newspapers1 = analyze_newspaper(zipfile1)
search_for_text("Christopher", newspapers1, 1.3)