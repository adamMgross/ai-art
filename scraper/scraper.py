from BeautifulSoup import BeautifulSoup
from urllib2 import urlopen
from urllib import unquote
import re
from subprocess import call
import sys
from time import sleep

DEBUG = False
try:
    if sys.argv[1]:
        DEBUG = True
except:
    pass




def sh_exec(s):
    call(s.split(' '))


def parse_page(url, genre):
    artist = url.split('/')[-1]
    print artist
    count = 0
    for src in BeautifulSoup(urlopen(url)).findAll('img'):
        if '%3A%2F%2F' in str(src):
            sleep(0.5)
            print extract_img_src(str(src))
            sh_exec('wget {} -O ./imgs/{}/{}-{}.jpg'
                    .format(extract_img_src(str(src)),
                            genre,
                            artist,
                            count))
            count += 1
            #print extract_img_src(str(src)), artist

def log(s):
    if DEBUG:
        for item in s:
            print s

def extract_img_src(img_tag):
    a = img_tag.split('https%3A%2F%2F')
    b = a[1]
    c = b.split('\"')
    d = c[0]
    log([a, b, c, d])
    return 'https://' + unquote(img_tag.split('https%3A%2F%2F')[1].split('\"')[0])

def download_art_by_genre():
    surreal_list = [url.strip() for url in open('./surrealist.txt', 'r').readlines()]
    #impression_list = [url.strip() for url in open('./impressionist.txt', 'r').readlines()]

    for artist in surreal_list:
        parse_page(artist, 'surrealist')
    #for artist in impression_list:
    #    parse_page(artist, 'impressionist')


download_art_by_genre()

