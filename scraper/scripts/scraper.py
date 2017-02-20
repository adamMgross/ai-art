from BeautifulSoup import BeautifulSoup
from urllib2 import urlopen
from urllib import unquote
import re
from subprocess import call
import sys
from time import sleep
import os.path

DEBUG = False
try:
    if sys.argv[1]:
        DEBUG = True
except:
    pass


def sh_exec(s):
    call(s.split(' '))

    
def parse_page(url, genre):
    try:
        artist = url.split('/')[-1]
        count = 0
        fail_count = 0
        for src in BeautifulSoup(urlopen(url)).findAll('img'):
            if '%3A%2F%2F' in str(src):
                #sleep(0.5)
                link = extract_img_src(str(src))
                try:
                    path = '../imgs/{}/{}-{}.jpg'.format(genre, artist, count)
                    if not os.path.isfile(path):
                        sh_exec('wget {} -O {}'.format(link, path))
                        count += 1
                except:
                    fail_count += 1
                    print 'FAILED: {}\n\tfrom {}'.format(link, artist)
        return (count, fail_count)
    except:
        print 'FAILED_URL: {}'.format(url)
        sleep(10)

        
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
    impression_list = [url.strip() for url in open('./impressionist.txt', 'r').readlines()]
    total_num_success = 0
    total_num_failed = 0
    for artist in surreal_list:
        if not artist:
            continue
        num_success, num_failed = parse_page(artist, 'surrealist')
        total_num_success += num_success
        total_num_failed += num_failed
        sh_exec('clear')
        print('Finished {}'.format(artist.split('/')[-1]))
        print 'SUCCESS: {}\nFAILED: {}'.format(total_num_success, total_num_failed)
        #sleep(0.5)
    for artist in impression_list:
        num_success, num_failed = parse_page(artist, 'impressionist')
        total_num_success += num_success
        total_num_failed += num_failed
        sh_exec('clear')
        print('Finished {}'.format(artist.split('/')[-1]))
        print 'SUCCESS: {}\nFAILED: {}'.format(total_num_success, total_num_failed)
    print 'SUCCESS: {}\nFAILED: {}'.format(total_num_success, total_num_failed)


download_art_by_genre()

