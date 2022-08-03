import time
import requests
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict

'''
Original Code
https://github.com/ssut/py-hanspell
'''


def _remove_tags(text):
    text = u'<content>{}</content>'.format(text).replace('<br>','')

    result = ''.join(ET.fromstring(text).itertext())

    return result

def spell_check(text):
    payload = {
            '_callback': 'window.__jindo2_callback._spellingCheck_0',
            'q': text
        }

    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        'referer': 'https://search.naver.com/',
    }

    base_url = 'https://m.search.naver.com/p/csearch/ocontent/spellchecker.nhn'

    _agent = requests.Session()
    start_time = time.time()
    r = _agent.get(base_url, params=payload, headers=headers)
    passed_time = time.time() - start_time

    r = r.text[42:-2]

    data = json.loads(r)
    html = data['message']['result']['html']
    result = {
        'result': True,
        'original': text,
        'checked': _remove_tags(html),
        'errors': data['message']['result']['errata_count'],
        'time': passed_time,
        'words': OrderedDict(),
    }

    html = html.replace('<span class=\'green_text\'>', '<green>') \
                .replace('<span class=\'red_text\'>', '<red>') \
                .replace('<span class=\'purple_text\'>', '<purple>') \
                .replace('<span class=\'blue_text\'>', '<blue>') \
                .replace('</span>', '<end>')
    items = html.split(' ')
    words = []
    tmp = ''
    for word in items:
        if tmp == '' and word[:1] == '<':
            pos = word.find('>') + 1
            tmp = word[:pos]
        elif tmp != '':
            word = u'{}{}'.format(tmp, word)

        if word[-5:] == '<end>':
            word = word.replace('<end>', '')
            tmp = ''

        words.append(word)

    result = []
    for word in words:
        if word[:5] == '<red>':
            word = word.replace('<red>', '')
        elif word[:7] == '<green>':
            word = word.replace('<green>', '')
        elif word[:8] == '<purple>':
            word = word.replace('<purple>', '')
        elif word[:6] == '<blue>':
            word = word.replace('<blue>', '')
        result.append(word)


    return ' '.join(result)