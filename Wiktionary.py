#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Libraries to analyze Wiktionary data dumps

The regex do not handle nested tags, which do exist on Wiktionary.
This needs to be corrected.
"""
import os
from pathlib import Path
from datetime import datetime
import pickle
import time
from tqdm import tqdm
import bz2
import re
import xml.etree.ElementTree as etree


class wiki_clean():
    """Class of functions and regular expressions to analyse Wiktionary
    data dump, including unwiki()
    """

    def __init__(self):
        # Regular expression for tags cleaning
        self.RE_wiki = re.compile('|'.join(
            [r"""\[\[(File|Category):[\s\S]+\]\]""",
             r"""\[\[[^|^\]]+\|""",
             r"""\[\[""",
             r"""\]\]""",
             r"""\'{2,5}""",
             r"""(<s>|<!--)[\s\S]+(</s>|-->)""",
             r"""{{[\s\S\n]+?}}""",
             r"""<ref [\s\S]+?</ref>""",
             r"""<[\s\S]+?>""",
             r"""={1,6}"""]
            ), re.VERBOSE)

        # Wiki tags where you keep everything after the first '|'  character,
        # the tag (or "head") being ignored in the cleaning
        label_list_keepfirall = {'non-gloss[ ]definition',
                                 'n-g',
                                 'non-gloss',
                                 'non[ ]gloss',
                                 'ngd'}

        # Wiki tags where you keep everything after the second '|' character
        # the tag (or "head") and the required first field
        # being ignored in the cleaning
        label_list_keepsecall = {'lb',
                                 'lbl',
                                 'label',
                                 'm',
                                 'mention',
                                 'l-self',
                                 'm-self',
                                 'll'}

        # Wiki tags where you keep everything before the second '|' character
        # the tag and the first field being kept in the cleaning
        label_list_keepheadfir = {'[^-]+of',
                                  'en-[\\s\\S]+of',
                                  'second-[\\s\\S]+of',
                                  'eye-[\\s\\S]+of',
                                  'alt[ ]form',
                                  'altform'}

        # Wiki tags where you keep everything including the tag
        label_list_keepheadall = {'surname'}

        # Wiki tags where you only keep the first field after the tag
        # what's between the first and the second '|' character
        label_list_keepfir = {'ws',
                              'taxlink'}

        # Wiki tags where you only keep the second field after the tag
        # what's between the second and the third '|' character
        label_list_keepsec = {'l',
                              'link'}

        # Putting the tags in the appropriate regular expressions
        self.RE_label_id_firall = re.compile('|'.join(
                [r"""{{%s\|[\s\S\n]+?}}""" % lbl
                 for lbl in label_list_keepfirall]
                ), re.VERBOSE)
        self.RE_label_id_secall = re.compile('|'.join(
                [r"""{{%s\|[\s\S\n]+?}}""" % lbl
                 for lbl in label_list_keepsecall]
                ), re.VERBOSE)
        self.RE_label_id_headfir = re.compile('|'.join(
                [r"""{{%s\|[\s\S\n]+?}}""" % lbl
                 for lbl in label_list_keepheadfir]
                ), re.VERBOSE)
        self.RE_label_id_headall = re.compile('|'.join(
                [r"""{{%s\|[\s\S\n]+?}}""" % lbl
                 for lbl in label_list_keepheadall]
                ), re.VERBOSE)
        self.RE_label_id_fir = re.compile('|'.join(
                [r"""{{%s\|[\s\S\n]+?}}""" % lbl
                 for lbl in label_list_keepfir]
                ), re.VERBOSE)
        self.RE_label_id_sec = re.compile('|'.join(
                [r"""{{%s\|[\s\S\n]+?}}""" % lbl
                 for lbl in label_list_keepsec]
                ), re.VERBOSE)

        # Regular expression for the sense tag
        self.RE_get_sense = re.compile(r"""{{sense\|[\s\S]+?}}""",
                                       re.VERBOSE)

        # Regular expression for the link tag
        self.RE_get_link = re.compile(
            r"""{{l\|[\s\S]+?}}|{{link\|[\s\S]+?}}""",
            re.VERBOSE
            )

        # Regular expression for the words/ws tag
        self.RE_get_ws = re.compile(r"""{{ws\|[\s\S]+?}}""", re.VERBOSE)

        # Regular expression for the synonyms tag
        self.RE_get_syn = re.compile(
            r"""{{syn\|[\s\S]+?}}|{{synonyms\|[\s\S]+?}}""",
            re.VERBOSE
            )

        # Regular expression for the Thesaurus tag
        self.RE_thesaurus = re.compile(r"""\[\[Thesaurus:[\s\S]+?\]\]""",
                                       re.VERBOSE)

        self.PARTS_OF_SPEECH = {
            "noun", "verb", "adjective", "adverb", "determiner",
            "article", "preposition", "conjunction", "proper noun",
            "letter", "character", "phrase", "proverb", "idiom",
            "symbol", "syllable", "numeral", "initialism", "interjection",
            "definitions"
        }

        self.RELATIONS = [
            "synonyms", "antonyms", "hypernyms", "hyponyms",
            "meronyms", "holonyms", "troponyms", "related terms",
            "derived terms", "coordinate terms"
        ]

        self.UNWANTED_LIST = [
            'External links', 'Compounds',
            'Anagrams', 'References', "Further reading"
            'Statistics', 'See also', 'Usage notes'
        ]

        self.LANG_TAGS = [
            'en', 'mg', 'fr', 'ru', 'sh', 'es', 'zh', 'de', 'nl', 'sv', 'ku',
            'pl', 'lt', 'el', 'it', 'fi', 'ca', 'ta', 'hu', 'tr', 'ko', 'io',
            'kn', 'hy', 'pt', 'vi', 'sr', 'ja', 'chr', 'hi', 'th', 'ro', 'no',
            'id', 'ml', 'et', 'my', 'uz', 'li', 'or', 'te', 'cs', 'fa', 'eo',
            'ar', 'jv', 'az', 'eu', 'gl', 'oc', 'da', 'br', 'lo', 'uk', 'hr',
            'fj', 'tg', 'bg', 'ky', 'simple', 'ps', 'ur', 'sk', 'cy', 'vo',
            'la', 'wa', 'is', 'zh-min-nan', 'af', 'scn', 'ast', 'he', 'tl',
            'sw', 'fy', 'nn', 'pa', 'lv', 'bn', 'co', 'mn', 'pnb', 'ka', 'nds',
            'sl', 'sq', 'lb', 'bs', 'nah', 'sa', 'kk', 'tk', 'km', 'sm', 'mk',
            'hsb', 'be', 'ms', 'ga', 'an', 'wo', 'vec', 'ang', 'tt', 'sd',
            'mt', 'gn', 'mr', 'ie', 'so', 'csb', 'ug', 'gd', 'st', 'roa-rup',
            'si', 'hif', 'ia', 'mi', 'ay', 'kl', 'fo', 'jbo', 'ln', 'zu', 'na',
            'gu', 'gv', 'kw', 'rw', 'ts', 'ne', 'om', 'qu', 'su', 'ss', 'ha',
            'iu', 'am', 'dv', 'za', 'tpi', 'ik', 'yi', 'ti', 'sg', 'tn', 'ks',
            'as', 'mo', 'pi', 'als', 'ab', 'sn', 'bh', 'dz', 'tw', 'to', 'bi',
            'yo', 'bo', 'rm', 'xh', 'aa', 'sc', 'bm', 'ak', 'cr', 'av', 'ch',
            'rn', 'mh'
        ]

    def clean_label_tag_firall(self,
                               matchobj):
        """
        Function to apply to tags where you keep everything after
        the first field, the tag header

        Parameters
        ----------
        matchobj : re match object
            object returned through the re.sub function, where match.group(0)
            is the result of the regular expression match

        Returns
        -------
        str
            cleaned tag string
        """
        return('(' +
               ', '.join([s for s in matchobj.group(0)[2:-2].split('|')[1:]
                          if '=' not in s and s not in {'', ' '}]) +
               ')')

    def clean_label_tag_secall(self,
                               matchobj):
        """
        Function to apply to tags where you keep everything after the first two
        fields.

        Parameters
        ----------
        matchobj : re match object
            object returned through the re.sub function, where match.group(0)
            is the result of the regular expression match

        Returns
        -------
        str
            cleaned tag string
        """
        return('(' +
               ', '.join([s for s in matchobj.group(0)[2:-2].split('|')[2:]
                          if '=' not in s and s not in {'', ' '}]) +
               ')')

    def clean_label_tag_headfir(self,
                                matchobj):
        """
        Function to apply to tags where you keep the first tag field
        (the header) and the second tag

        Parameters
        ----------
        matchobj : re match object
            object returned through the re.sub function, where match.group(0)
            is the result of the regular expression match

        Returns
        -------
        str
            cleaned tag string
        """
        string = ' '.join(matchobj.group(0)[2:-2].split('|')[:2])
        string = string.replace('alt_form', 'Alternative form of')
        string = string.replace('alt form', 'Alternative form of')
        string = string.replace('alternate form', 'Alternative form')
        string = string.replace('en-', '')
        return '[' + string + ']'

    def clean_label_tag_headall(self,
                                matchobj):
        """
        Function to apply to tags where you keep every field,
        including the tag header

        Parameters
        ----------
        matchobj : re match object
            object returned through the re.sub function, where match.group(0)
            is the result of the regular expression match

        Returns
        -------
        str
            cleaned tag string
        """
        return(' '.join(
            [s for s in reversed(matchobj.group(0)[2:-2].split('|'))
             if '=' not in s and s not in {'', ' '}]
            ))

    def clean_label_tag_fir(self,
                            matchobj):
        """
        Function to apply to tags where you only keep the second tag field,
        which is the first after the tag header

        Parameters
        ----------
        matchobj : re match object
            object returned through the re.sub function, where match.group(0)
            is the result of the regular expression match

        Returns
        -------
        str
            cleaned tag string
        """
        return matchobj.group(0)[2:-2].split('|')[1].replace(' ', '_')

    def clean_label_tag_sec(self,
                            matchobj):
        """
        Function to apply to tags where you keep the third tag field,
        which is the second after the tag header.

        Parameters
        ----------
        matchobj : re match object
            object returned through the re.sub function, where match.group(0)
            is the result of the regular expression match

        Returns
        -------
        str
            cleaned tag string
        """
        return matchobj.group(0)[2:-2].split('|')[2].replace(' ', '_')

    def unwiki(self,
               text):
        "Clean text of wikipedia tags"
        result = re.sub(self.RE_label_id_firall,
                        self.clean_label_tag_firall,
                        text)
        result = re.sub(self.RE_label_id_secall,
                        self.clean_label_tag_secall,
                        result)
        result = re.sub(self.RE_label_id_headfir,
                        self.clean_label_tag_headfir,
                        result)
        result = re.sub(self.RE_label_id_headall,
                        self.clean_label_tag_headall,
                        result)
        result = re.sub(self.RE_label_id_fir,
                        self.clean_label_tag_fir,
                        result)
        result = re.sub(self.RE_label_id_sec,
                        self.clean_label_tag_sec,
                        result)
        result = re.sub(self.RE_label_id_sec,
                        self.clean_label_tag_sec,
                        result)
        result = re.sub(self.RE_wiki,
                        '',
                        result)
        return re.sub(r' +', ' ', result)

    def get_sense(self,
                  text):
        """Gets list of senses in the sense tag for synonyms section

        Parameters
        ----------
        text: str
            Text from a wiktionary page

        Returns
        -------
        list of str
            List of senses
        """
        return [self.unwiki(sense.replace('{{sense|', '').replace('}}', ''))
                for sense in self.RE_get_sense.findall(text)]

    def clean_link(self,
                   text):
        """Cleans the link text from a link tag

        Parameters
        ----------
        text: str
            The whole link template tag

        Returns
        -------
        str
            The cleaned link tag, corresponding to the third
            field of the tag (the one after the header and the language)
        """
        return('_'.join(text[2:-2].split('|')[2].split()))

    def get_link(self,
                 text):
        """Gets list of links/words for synonyms section

        Parameters
        ----------
        text: str
            Text from a wiktionary page

        Returns
        -------
        list of str
            List of links/words
        """
        return [self.clean_link(link)
                for link in self.RE_get_link.findall(text)]

    def clean_ws(self,
                 text):
        """Cleans the text from a words/ws tag

        Parameters
        ----------
        text: str
            The whole words/ws template tag

        Returns
        -------
        str
            The cleaned words/ws tag, corresponding to the first
            field of the tag (the one after the header and the language)
        """
        return('_'.join(text[2:-2].split('|')[1].split()))

    def get_ws(self,
               text):
        """Gets list of ws/words for Thesaurus pages

        Parameters
        ----------
        text: str
            Text from a wiktionary page

        Returns
        -------
        list of str
            List of words
        """
        return [self.clean_ws(ws)
                for ws in self.RE_get_ws.findall(text)]

    def clean_syn(self,
                  text):
        """Cleans the text from a synonyms tag

        Parameters
        ----------
        text: str
            The whole synonyms template tag

        Returns
        -------
        str
            The cleaned synonyms tag, corresponding to the first
            field of the tag (the one after the header and the language)
        """
        return ['_'.join(s.split())
                for s in text[2:-2].split('|')[2:]
                if '=' not in s and s not in {'', ' '}]

    def get_syn(self,
                text):
        """Gets list of synonyms in the definition sections

        Parameters
        ----------
        text: str
            Text from a wiktionary page

        Returns
        -------
        list of str
            List of senses
        """

        return [syn for syns in self.RE_get_syn.findall(text)
                for syn in self.clean_syn(syns)]

    def wiktionary_info(self,
                        text,
                        languages=None,
                        parts_of_speech=None):
        """Extract information (definition and synonyms) from a Wiktionary page

        Parameters
        ----------
        text: str
            The raw text of a Wiktionary page from an xml data dump

        languages: set of str, optional
            The languages to consider. The languages tags are the section
            titles in the page itself. If None, will consider all languages.
            Default is None

        parts_of_speech: set of str, optional
            The parts of speech, or pos (noun, verb, ...) to consider.
            The pos tags are the sections titles in the page itself.
            If None, will consider all possible pos.
            Default is None

        Returns
        -------
        definitions, synonyms, nested_synonyms
            Definitions will contain all definitions of the word,
            a single definition being a string

            Synonyms will contain all word tokens which are synonyms of the
            worda single word token being a string. Some of these word tokens
            will be of the form Thesaurus:[word token]

            Nested_synonyms will contain all word tokens whose synonyms
            present in a Thesaurus:[word token] page should be added
            to the current word's synonyms list.

            Each object separates all items by language and pos
            in a structure dictionary. For example, to get the list of all
            definitions for the english language and with the word considered
            as a noun, check definitions['english']['noun']

            definitions: {language (str):
                              {pos (str):
                                   [definition (str)]
                               }
                          }

            synonyms: {language (str):
                           {pos (str):
                                [definition (str)]
                            }
                       }

            nested_synonyms: {language (str):
                                  {pos (str):
                                       [definition (str)]
                                   }
                              }
        """

        definitions = {}
        synonyms = {}
        nested_synonyms = {}
        if languages:
            languages = {lang.lower() for lang in languages}
        if parts_of_speech:
            parts_of_speech = {pos.lower() for pos in parts_of_speech}
        lang = ''
        pos = ''
        rel = ''
        for line in text.split('\n'):
            # This code gives the contents table location of the current text
            # but only if it's part of previously defined relevant set
            # If not in the set or unknown, will be an empty string ''

            # lvl: number of '=' characters at the start/end of the line
            # tag: tring in between the '=' characters

            # lang: current language of the section.
            # a language section is defined between two equals,
            # i.e. '==English=='

            # pos: part-of-speech of the current section.
            # A pos section is defined between three equals,
            # i.e. '===Noun==='

            # rel: related sub-section under the part-os-speech
            # Those sections are defined between four equals,
            # i.e. '====Synonyms===='
            # This is only used to track synonyms

            lvl = 0
            tag = ''
            while (lvl < len(line)//2 and
                   line[lvl] == '=' and
                   line[-1-lvl] == '='):
                lvl += 1
            if lvl != 0:
                tag = line[lvl:-lvl]
            if lvl == 2 and (not languages or
                             tag.lower() in languages):
                lang = tag.lower()
            elif lvl <= 2 and lvl != 0:
                lang = ''
            if lang != '':
                if lvl >= 3 and (not parts_of_speech or
                                 tag.lower() in parts_of_speech):
                    pos = tag.lower()
                elif lvl <= 3 and lvl != 0:
                    pos = ''
                if lvl >= 4 and tag.lower() == 'synonyms':
                    rel = 'synonyms'
                elif lvl != 0:
                    rel = ''

            # Treatment of definition section
            if lvl == 0 and lang != '' and pos != '' and line != '':
                hashlvl = 0
                while hashlvl < len(line)-2 and line[hashlvl] == '#':
                    hashlvl += 1
                if line[hashlvl] == ' ':
                    if lang not in definitions:
                        definitions[lang] = {}
                    unwiki_data = self.unwiki(line[hashlvl+1:])
                    if unwiki_data != '':
                        if pos not in definitions[lang]:
                            definitions[lang][pos] = [unwiki_data]
                        else:
                            definitions[lang][pos].append(unwiki_data)

                # Sometimes, synonyms are in the definition section
                # under a synonym tag
                for syn in self.get_syn(line):
                    if lang not in synonyms:
                        synonyms[lang] = {}
                    if pos not in synonyms[lang]:
                        synonyms[lang][pos] = [syn]
                    elif syn not in synonyms[lang][pos]:
                        synonyms[lang][pos].append(syn)

            # Synonyms, and sense in definitions, and Thesaurus linking
            if (lvl == 0 and
                    lang != '' and
                    pos != '' and
                    line != '' and
                    rel == 'synonyms'):

                # hashlvl: Number of star '*' characters at the start
                # of the line

                hashlvl = 0
                while hashlvl < len(line)-2 and line[hashlvl] == '*':
                    hashlvl += 1

                if line[hashlvl] == ' ':
                    # Link tags are counted as synonyms
                    for link in self.get_link(line[hashlvl+1:]):
                        if lang not in synonyms:
                            synonyms[lang] = {}
                        if pos not in synonyms[lang]:
                            synonyms[lang][pos] = [link]
                        elif link not in synonyms[lang][pos]:
                            synonyms[lang][pos].append(link)
                    # Sense tags are counted as synonyms
                    for sense in self.get_sense(line[hashlvl+1:]):
                        if lang not in definitions:
                            definitions[lang] = {}
                        if pos not in definitions[lang]:
                            definitions[lang][pos] = [sense]
                        else:
                            definitions[lang][pos].append(sense)
                # Words/ws tags are counted as synonyms
                for ws in self.get_ws(line):
                    if lang not in synonyms:
                        synonyms[lang] = {}
                    if pos not in synonyms[lang]:
                        synonyms[lang][pos] = [ws]
                    elif ws not in synonyms[lang][pos]:
                        synonyms[lang][pos].append(ws)

                # Thesaurus tag. Will link to another page including
                # synonyms of the current word. The title
                # of the page is saved in nested_synonyms, in order to
                # include later the synonyms of that page to the current one
                for thes in self.RE_thesaurus.findall(line):
                    cleanthes = thes.replace('[[', '').replace(']]', '')
                    cleanthes = cleanthes.split('#')[0].split('|')[0]
                    if lang not in nested_synonyms:
                        nested_synonyms[lang] = {}
                    if pos not in nested_synonyms[lang]:
                        nested_synonyms[lang][pos] = [cleanthes]
                    elif cleanthes not in nested_synonyms[lang][pos]:
                        nested_synonyms[lang][pos].append(cleanthes)

        return definitions, synonyms, nested_synonyms


class Definition():
    """Word with definition, synonyms and semantic information

    Parameters
    ----------
    word: str
        The word token associated to the definition object

    definitions: dict of dict of dict of list of str
        Definitions will contain all definitions of the word,
        a single definition being a string

        Each object separates all items by language and pos
        in a structure dictionary. For example, to get the list of all
        definitions for the english language and with the word considered
        as a noun, check definitions['english']['noun']

        {language (str):
             {pos (str):
                  [definition (str)]
              }
         }

    synonyms:
        Synonyms will contain all word tokens which are synonyms of the
        worda single word token being a string.

        Each object separates all items by language and pos
        in a structure dictionary. For example, to get the list of all
        synonyms for the english language and with the word considered
        as a noun, check synonyms['english']['noun']

        {language (str):
             {pos (str):
                  [definition (str)]
              }
         }
    """

    def __init__(self,
                 word,
                 definitions=None,
                 synonyms=None):
        self.word = word
        if definitions is None:
            self.definitions = {}
        else:
            self.definitions = definitions
        if synonyms is None:
            self.synonyms = {}
        else:
            self.synonyms = synonyms

    def update_syn(self,
                   lang,
                   pos,
                   new_syns):
        """Updates synonyms object for a specific language and part-of-speech

        Parameters
        ----------
        lang : str
            Language of the word token to consider.

        pos : str
            Part of speech of the word token to consider.

        new_syns : list of str
            list of the word tokens of synonyms to add to the synonym object.
        """
        if lang not in self.synonyms:
            self.synonyms[lang] = {}
        if pos not in self.synonyms[lang]:
            self.synonyms[lang][pos] = new_syns
        else:
            self.synonyms[lang][pos] = \
                list(set(self.synonyms[lang][pos] + new_syns))


def fixtag(ns, tag, nsmap):
    """Fixes the XML tag based on the nsmap dictionary

    Parameters
    ----------
    ns: str
        Namespace of the xml file

    tag: str
        the XML tag to be corrected with the right prefix

    nsmap: dict of str:str
        the namespace map, where the keys as the namespaces
        and the value is the correct prefix
    """
    return '{' + nsmap[ns] + '}' + tag


def parse_wiktionary_dump():
    """Parses through the wiktionary data dump and extracts relevant info

    Will create 2 pickle objects.

    Wiktionary_data: List of all Definitions() objects

    Wikitionary_dict: Dictionary where key=word_token and value=index of the
    corresponding Definition() object in the Wiktionary_data list
    """
    # Unzips wiktionary dump file, if needed be
    datapath = Path() / 'data'
    zip_names = []
    unzip_names = []

    for f in os.listdir(datapath):
        fs = f.split('-')
        if (len(fs) == 4 and
                fs[0] == 'enwiktionary' and
                fs[2] == 'pages'):
            if fs[3] == 'articles.xml.bz2':
                zip_names.append((f, int(fs[1])))
            if fs[3] == 'articles.xml':
                unzip_names.append((f, int(fs[1])))
    zip_names = sorted(zip_names, key=lambda x: -x[1])
    unzip_names = sorted(unzip_names, key=lambda x: -x[1])
    if not unzip_names or (zip_names and zip_names[0][1] > unzip_names[0][1]):
        if not zip_names:
            raise NameError('No wiktionary dump found in ./data/. '
                            'Please download from '
                            'https://dumps.wikimedia.org/enwiktionary/ '
                            ', click on the most recent date and download '
                            'the file '
                            'enwiktionary-XXXXXXXX-pages-articles.xml.bz2 '
                            '(no multistream)')

        print(datetime.now(), 'Unzipping .bz2 file')

        zip_path = datapath / zip_names[0][0]
        # XML dump file
        unzip_path = datapath / (zip_names[0][0][:-4])

        with open(unzip_path, 'wb') as new_file, open(zip_path, 'rb') as file:
            decompressor = bz2.BZ2Decompressor()
            for data in iter(lambda: file.read(100 * 1024), b''):
                new_file.write(decompressor.decompress(data))
    else:
        unzip_path = datapath / unzip_names[0][0]

    WC = wiki_clean()

    # IMPORT DEFINITIONS

    # List of all definitions objects
    Wiktionary_data = []
    # Dictionary where key=word_token and value=index of the definition
    # in Wiktionary_data list
    Wiktionary_dict = {}
    # Dictionary where key=Thesaurus:[word_token] and
    # value=list of synonyms of the word
    # Taken from Thesaurus:[] pages
    Wiktionary_thes = {}
    # Dictionary where key=word_token and
    # value=list of 'Thesaurus:[word_token]'
    # of which the synonyms on their Thesaurus page should be added to the
    # current word token. The
    Wiktionary_nested = {}

    print(str(datetime.now()), 'Start parsing through Wiktionary data')
    time.sleep(0.5)
    start_time = time.time()
    i = 0
    # XML namespace map
    nsmap = {}
    # List of all word tokens
    titles = []
    redirects = {}

    # 7.00M pages (2020-09)
    with tqdm(total=7003907) as pbar:
        for event, elem in etree.iterparse(
                unzip_path,
                events=('start', 'end', 'start-ns', 'end-ns')
                ):
            if event == 'start-ns':
                ns, url = elem
                nsmap[ns] = url
            if event == 'end' and elem.tag == fixtag('', 'page', nsmap):
                i += 1
                pbar.update()

                # Title
                title = ''
                # Redirect
                red = ''
                definitions = {}
                synonyms = {}
                nested_synonyms = {}
                for child in elem:
                    # Title is the word token
                    if child.tag == fixtag('', 'title', nsmap):
                        title = child.text
                        titles.append(title)

                    # Will contain the definition of the word
                    if child.tag == fixtag('', 'revision', nsmap):
                        for child2 in child:
                            if (child2.tag == fixtag('', 'text', nsmap) and
                                    child2.text is not None and
                                    '==English==' in child2.text):
                                definitions, synonyms, nested_synonyms = \
                                    WC.wiktionary_info(child2.text,
                                                       languages={'English'})

                    # If page is a redirect page, will be added as a synonym
                    if child.tag == fixtag('', 'redirect', nsmap):
                        red = child.attrib['title']

                if title and red:
                    redirects[title] = red

                if (title and ':' not in title and
                        (sum([len(v) for v in definitions.values()]) != 0 or
                         len(synonyms) != 0)):
                    # If more than one page with the same title
                    if title in Wiktionary_dict:
                        raise NameError(title, 'CLONED')

                    Wiktionary_dict[title] = len(Wiktionary_data)
                    for lang, d in synonyms.items():
                        for pos, d2 in d.items():
                            synonyms[lang][pos] = list(set(
                                [s.replace('Thesaurus:', '')
                                 for s in d2]
                                ))
                    Wiktionary_data.append(Definition(title,
                                                      definitions,
                                                      synonyms))
                    if len(nested_synonyms) != 0:
                        Wiktionary_nested[title] = nested_synonyms
                # If a thesaurus page
                if title != '' and title[:10] == 'Thesaurus:':
                    # If more than one Thesaurus page with the same title
                    if title in Wiktionary_thes:
                        raise NameError(title, 'CLONED')
                    Wiktionary_thes[title] = synonyms

                # Freeing memory linked to pages data
                elem.clear()

    print()
    print('Scrape took %.1f sec' % (time.time()-start_time))

    print(str(datetime.now()), 'Adding Thesaurus synonyms')
    n = 0
    # If mention of a Thesaurus page in the definition page
    for word, d in Wiktionary_nested.items():
        if word not in Wiktionary_dict:
            raise NameError('word not in Wiktionary_dict, it should be')
        word_data = Wiktionary_data[Wiktionary_dict[word]]
        for lang, d2 in d.items():
            for pos, thestags in d2.items():
                for thestag in thestags:
                    if (thestag in Wiktionary_thes and
                            lang in Wiktionary_thes[thestag] and
                            pos in Wiktionary_thes[thestag][lang]):

                        # Listing new synonyms to add
                        new_syns = list(set(
                            [s.replace('Thesaurus:', '')
                             for s in Wiktionary_thes[thestag][lang][pos]]
                            ))

                        # Counting added synonyms
                        if (lang in word_data.synonyms and
                                pos in word_data.synonyms[lang]):
                            for s in new_syns:
                                if s not in word_data.synonyms[lang][pos]:
                                    n += 1
                        else:
                            n += len(new_syns)

                        # Adding new synonyms
                        word_data.update_syn(lang, pos, new_syns)

    # If mentioned by the Thesaurus page itself, to its title
    for thestag, d in Wiktionary_thes.items():
        word = thestag.replace('Thesaurus:', '')
        if word not in Wiktionary_dict:
            continue
        word_data = Wiktionary_data[Wiktionary_dict[word]]
        for lang, d2 in d.items():
            for pos, new_syns in d2.items():
                if word in Wiktionary_dict:

                    # Listing new synonyms to add
                    new_syns = list(set(
                        [s.replace('Thesaurus:', '')
                         for s in new_syns]
                        ))

                    # Counting added synonyms
                    if (lang in word_data.synonyms and
                            pos in word_data.synonyms[lang]):
                        for s in set(new_syns):
                            if s not in word_data.synonyms[lang][pos]:
                                n += 1
                    else:
                        n += len(new_syns)

                    # Adding new synonyms
                    word_data.update_syn(lang, pos, new_syns)
    print(f'{n} synonyms added through Thesaurus nesting')

    print(str(datetime.now()), 'Adding redirection pages data')
    n = 0
    for syn, word in redirects.items():
        if word not in Wiktionary_dict:
            continue
        word_data = Wiktionary_data[Wiktionary_dict[word]]
        Wiktionary_dict[syn] = Wiktionary_dict[word]
        for lang in word_data.definitions:
            for pos in word_data.definitions[lang]:

                # Counting added synonyms
                if (lang in word_data.synonyms and
                        pos in word_data.synonyms[lang]):
                    if syn not in word_data.synonyms[lang][pos]:
                        n += 1
                else:
                    n += 1

                word_data.update_syn(lang,
                                     pos,
                                     [syn])

    print(f'{n} synonyms added through redirect pages')

    print('Saving data')
    pickle.dump(Wiktionary_data,
                open(datapath / 'Wiktionary_data.pkl', 'wb'))
    pickle.dump(Wiktionary_dict,
                open(datapath / 'Wiktionary_dict.pkl', 'wb'))

    print(str(datetime.now())+'\t'+'Parsing through Wiktionary data done')


class wiktionary():
    """Wiktionary object containing relevant function for semantic
    word embedding traning

    parse_wiktionary_dump() must be run first to create the .pkl objects

    Parameters
    ----------
    path: str, optional
        Path of the model files Wiktionary_data.pkl and Wiktionary_dict.pkl
        Default is None, which default to Path() / 'data'

    Attributes
    ----------
    data: List of all Definitions() objects

    dict: Dictionary where key=word_token and value=index of
        the corresponding Definition() object in the Wiktionary_data list
    """

    def __init__(self,
                 path=None):
        if path is None:
            path = Path() / 'data'
        else:
            path = Path(path)
        self.data = pickle.load(open(path / 'Wiktionary_data.pkl', 'rb'))
        self.dict = pickle.load(open(path / 'Wiktionary_dict.pkl', 'rb'))

    def worddata(self,
                 word):
        """Returns the Definition() object of a word token

        Parameters
        ----------
        word: str
            Work token

        Returns
        -------
        Definition() object
            The list of definitions and synonyms for the word token
            in all languages and part of speech.
        """
        if word not in self.dict:
            return None
        return self.data[self.dict[word]]

    def main_pos(self,
                 word,
                 lang='english'):
        """Returns the main part of speech (pos) for a word token in a
        specified language

        The main pos is the one who have the most definitions.

        Parameters
        ----------
        word: str
            Work token

        lang: str, optional
            Language to consider. Default is 'english'

        Returns
        -------
        str
            The main part of speech of the word token.
        """
        dat = self.worddata(word)
        if dat is None:
            return None

        all_defs = dat.definitions
        if lang not in all_defs:
            return None

        all_pos = [(pos, len(defs)) for pos, defs in all_defs[lang].items()]
        if not all_pos:
            return None

        all_pos = sorted(all_pos, key=lambda x: -x[1])
        return all_pos[0][0]


if __name__ == '__main__':
    # parse_wiktionary_dump()
    pass
