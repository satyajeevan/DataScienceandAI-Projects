# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 22:13:55 2021

@author: satya
"""

from flask import Flask,request,render_template
from flask_uploads import UploadSet,IMAGES
from google_trans_new import google_translator 
#from werkzeug.utils import secure_filename
#from werkzeug.datastructures import  FileStorage
#import easyocr
import os
import webbrowser
import re
from PIL import Image
import pytesseract
import spacy
# Load English tokenizer, tagger, parser and NER
#import en_core_web_sm
from nltk import word_tokenize 
import pandas as pd

app = Flask(__name__,static_url_path='',static_folder='static',template_folder='templates')

photos=UploadSet('photos',IMAGES)

app.config['UPLOAD_FOLDER']= 'IMAGES'  #our folder with collection of images created in the directory

#pytesseract
pytesseract.pytesseract.tesseract_cmd=r"C:/Program Files/Tesseract-OCR/tesseract.exe" #exe file for using ocr 

#baseurl
base_url='https://www.google.com/maps/search/'

#datasets
collect=pd.read_csv(r'collection.csv')
collect.columns=['postalcode',"placename","state1","country","community1","latitude","longitude"]


#translator
translator = google_translator() 

def preprocess_address(result_combine):
    postal_code=re.findall(r"\d{3}-\d{4}",result_combine)
    if postal_code:
        print("Pattern is correct - searching for address")
        if postal_code[0] in result_combine:
            postal_index=result_combine.index(postal_code[0])
        ls=result_combine[postal_index+8:]
    else:
        return None
	#translate text 
    translate_text = translator.translate(ls,lang_tgt='en')  
    print(translate_text)
    sp = spacy.load('en_core_web_sm')
    all_stopwords = sp.Defaults.stop_words
    text_tokens = word_tokenize(translate_text)
    tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]
    print(tokens_without_sw)
    #cleaning data
    clean_data=" ".join(tokens_without_sw)
    clean_data
    cities=pd.read_csv(r'D:\A.I\Japnese_translator_map\deployment\cities.csv')
    # new data frame with split value columns 
    new = cities['city_en'].str.split(",", n = 1, expand = True) 
    # making separate first name column from new data frame 
    cities["Special ward"]= new[0] 
    # making separate last name column from new data frame 
    cities["City"]= new[1] 
    # df display 
    lcities=list(cities.City.unique())
    sp=list(cities["Special ward"].unique())
    k=translate_text.split()
    city1=[x for x in sp if x in k]
    city2=[x for x in lcities if x in k]
    city1=" ".join(city1)
    city2=" ".join(city2)
    prefectures=pd.read_csv(r'prefectures.csv')
    new2= prefectures['prefecture_en'].str.split(" ", n = 1, expand = True) 
    # making separate first name column from new data frame 
    prefectures["pref"]= new2[0] 
    prefectures["name"]= new2[1] 
    pref_id=list(prefectures["pref"].unique())
    pref_id2=list(prefectures["name"].unique())
    pref=[x for x in pref_id if x in k]
    pref2=[x for x in pref_id2 if x in k]
    pref=" ".join(pref)
    pref22=" ".join(pref2)
    cut1=[str(pref),str(pref22),str(city2),str(city1)]
    part1join=" ".join(cut1)+" city"
    part1join
    rest=[]
    for word in tokens_without_sw:
        if word not in cut1:
            rest.append(word)
    #door_number_extract1=re.findall('(\d+)+-(\d+)',str(rest))
    print(str(rest)+"rest")
    door_number_extract=re.findall('[\d+]+-[\d+]+-[\d+]+|[\d+]+-[\d+]+|[\d+]+[-chome]+[\d+]|[\d+]+[-chome]+',str(rest))
    pattern=[" ".join(door_number_extract),part1join]
    #pattern=[",".join(door_number_extract),part1join]
    #fin=[part1join,"".join(DN)]
    Ready_browse=",".join(pattern)+","+postal_code[0]+'Japan'
    print(Ready_browse)
    return Ready_browse

@app.route("/",methods=['GET','POST'])
def home():
    if request.method == 'POST':
        if 'photo' not in request.files:
            return 'there is no photo'
        name=request.form['img-name']+'.jpeg' #img-name to be defined from html page
        photo=request.files['photo']
        lang=request.form['Language']
        print(lang)
        path=os.path.join(app.config['UPLOAD_FOLDER'],name)
        photo.save(path)#to save the delivery note of 1,2 3
        tex = pytesseract.image_to_string(Image.open(photo.filename),lang='jpn')
        result_sort=" ".join(tex.split())
        print(result_sort)
        #result_combine = re.sub(r"[A-Z][A-Za-z][A-Za-z{3}]\s.+[\d\.$]|[Tel].+[\d\.$]", " ", result_sort)
        result_combine = re.sub(r"[TEL].+[\d\.$]"," ",result_sort)
        print(result_combine)
        if lang == '0':
            #c = webbrowser.get('google-chrome')
            translator = google_translator()
            translate_text = translator.translate(result_combine,lang_tgt='en')  
            postal_code=re.findall(r"\d{3}-\d{4}",result_combine)
            for word in collect.postalcode:
                for code in postal_code:
                    if code == word:
                        k=collect.loc[collect['postalcode']==code]
                        final=k.iloc[:,:4].to_string(header=False,index=False,na_rep='NaN')
                        finallist=final.split()
                        rest=[]
                        for word in translate_text.split():
                            if word not in finallist:
                                rest.append(word)
                        door_number_extract=re.findall('[\d+]+-[\d+]+-[\d+]+|[\d+]+-[\d+]+|[\d+]+[-chome]+[\d+]|[\d+]+[-chome]+',str(rest))
                        if door_number_extract:
                            pattern=" ".join(door_number_extract)+","+final
                            webbrowser.open_new_tab(base_url+pattern)
                            return render_template("index.html",translate_text="\nPattern is correct - searching for address(Door Number/Town-City/Pincode):"+" \n "+pattern)
                        else:
                            break
                    else: 
                        break
            randomresult=preprocess_address(result_combine)
            if randomresult is None:
                return render_template("index.html",translate_text="Search completely failed- Please check database/Retake the image")
            else:
                finaddress=randomresult.replace(" ","")
                webbrowser.open_new_tab(base_url+finaddress)
                print(finaddress)
                return render_template("index.html",translate_text="Partial correct-Door Number/Database is not updated properly"+" \n "+finaddress)
        else:
            translator = google_translator()
            translate_text = translator.translate(result_combine,lang_tgt='en')  
            postal_code=re.findall(r"\d{3}-\d{4}",result_combine)
            for word in collect.postalcode:
                for code in postal_code:
                    if code == word:
                        k=collect.loc[collect['postalcode']==code]
                        final=k.iloc[:,:5].to_string(header=False,index=False,na_rep='NaN')
                        finallist=final.split()
                        rest=[]
                        for word in translate_text.split():
                            if word not in finallist:
                                rest.append(word)
                        door_number_extract=re.findall('[\d+]+-[\d+]+-[\d+]+|[\d+]+-[\d+]+|[\d+]+[-chome]+[\d+]|[\d+]+[-chome]+',str(rest))
                        if door_number_extract:
                            pattern=" ".join(door_number_extract)+","+final
                            translator = google_translator() 
                            translate_text = translator.translate(pattern,lang_tgt='ja')
                            webbrowser.open_new_tab(base_url+pattern)
                            return render_template("index.html",translate_text="\nPattern is correct - searching for address(Door Number/Town-City/Pincode):"+" \n "+translate_text)
                        else:
                            break
                    else: 
                        break
            randomresult=preprocess_address(result_combine)
            if randomresult is None:
                return render_template("index.html",translate_text="Search completely failed- Please check database/Retake the image")
            else:
                jap=randomresult.replace(" ","")
                translate_text_jp = translator.translate(jap,lang_tgt='ja')
                webbrowser.open_new_tab(base_url+translate_text)
                return render_template("index.html",translate_text="Database is not updated"+" \n "+translate_text_jp)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)