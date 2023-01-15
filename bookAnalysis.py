import requests
from bs4 import BeautifulSoup
import os
import PyPDF2
import fnmatch
from alive_progress import alive_bar
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import circlify
import seaborn as sns
import matplotlib.pyplot as plt
class Book_Analysis:
    """Automates data acquisition from engineering books using Web-Scraping 
    and desktop management. It also includes data preprocessing, datasets
    consolidation (pdf, text and figures), NLP processes, and building the 
    word-clouds plots. 
    """
    def __init__(self, url, pdf_dir, txt_dir, img_dir):
        """Default constructor

        Args:
            url (string): site to make the web-scraping process
            pdf_dir (string): folder to save PDF type files
            txt_dir (string): folder to save TXT converted from PDF type piles
            img_dir (string): folder to save word-cloud plots
        """
        self._url = url
        self._pdf_dir = pdf_dir
        self._txt_dir = txt_dir
        self._png_dir = img_dir
    def get_urls(self):
        """getting book's urls of the repository website
        """
        page = requests.get(self._url)
        soup = BeautifulSoup(page.content, 'html.parser')
        try:
            divs = soup.find_all('div', class_='bookContainer grow')
            with open(f'book_urls.txt', 'w') as fd:
                for div in divs:
                    book_url = div.findChild('a', href=True)['href']
                    download_url = self._url + book_url
                    download_page = requests.get(download_url)
                    download_soup = BeautifulSoup(download_page.content, 'html.parser')
                    footer = download_soup.find('div', {'id' : 'footer'})
                    file_name = footer.contents[0]
                    full_url = download_url + file_name
                    fd.write(full_url + '\n')
        except:
            print('something wrong')
        
    def to_text(self):
        """Extracting the text from the PDF Files
        """
        total = len(fnmatch.filter(os.listdir(self._pdf_dir),'*.pdf'))
        counter = 1
        for filename in os.listdir(self._pdf_dir):
            pdfFileObj = open(os.path.join(self._pdf_dir,filename), 'rb')
            try:
                pdfReader =PyPDF2.PdfFileReader(pdfFileObj)
                text_filename = filename.replace('.pdf','.txt')
                with open(os.path.join(self._txt_dir,text_filename), "w")as fd:
                    print(f'Book ,{counter}/{total}:{filename}...')
                    with alive_bar(pdfReader.numPages) as bar:
                        for page in range(pdfReader.numPages):
                            pageObj = pdfReader.getPage(page)
                            fd.write(pageObj.extractText() + '\n\n')
                            bar()
                    pdfFileObj.close()
                counter+=1
            except: 
                pass


    def get_relevant_words(self, word_count, doc):
        data = []
        total = len(doc)
        print(total)
        with alive_bar(total) as bar:
            for statement in doc:
                try:
                    clean_data = re.sub('[^a-zA-Z]', ' ', statement)
                    tokens = word_tokenize(clean_data)
                    myTokens = [word.lower() for word in tokens]
                    if len(myTokens):
                        statement = ''
                        for word in myTokens:
                            if not word in stopwords.words():
                                statement += word.lower() + ' '
                            if len(statement.strip()) > 0:
                                data.append(statement.strip())
                    bar()
                except (RuntimeError, TypeError, NameError):
                    print(RuntimeError.with_traceback())
                    print(TypeError.with_traceback())
                    print(NameError.with_traceback())
                    print(RuntimeError, TypeError, NameError)
        tr_idf_model = TfidfVectorizer(ngram_range=(1,1))
        tf_idf_vector = tr_idf_model.fit_transform(data)
        weights = [(word, tf_idf_vector.getcol(idx).sum()) for word , idx in tr_idf_model.vocabulary_.items()]
        weights.sort(key=lambda i:i[1], reverse=True)
        if len(weights) < word_count:
            return weights
        else:
            return weights[:word_count]

    def get_color_dict(self, palette, number, start):
        number = int(number * 100)
        pal = list(sns.color_palette(palette, n_colors=number).as_hex())
        return dict(enumerate(pal, start))

    def word_cloud(self, data, filename):
        df_words = pd.DataFrame(data, columns=['words', 'count'])
        circles = circlify.circlify(
            df_words['count'].tolist(),
            show_enclosure=False,
            target_enclosure=circlify.Circle(x=0, y=0)
        )
        n = df_words['count'].max()
        color_dict = self.get_color_dict(sns.color_palette("pastel"), n, 1)
        fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')
        ax.axis('off')
        lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r) for circle in circles)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        labels = list(df_words['words'])
        counts = list(df_words['count'])
        labels.reverse()
        counts.reverse()

        # mostrar circulos
        for circle, label, color in zip(circles, labels, counts):
            x, y, r = circle
            ax.add_patch(
                plt.Circle(
                    (x,y),
                    r,
                    alpha= 0.9,
                    linewidth=2,
                    color=color_dict.get(int(color * 100))
                )
            )
            plt.annotate(label, (x,y), size=12, va='center', ha='center')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(self._png_dir, filename.replace('.txt', '.png')))

    def word_freq(self):
        """_summary_
        """
        for filename in os.listdir(self._txt_dir):
            fullname = os.path.join(self._txt_dir, filename)
            if os.path.isfile(fullname):
                print(f'Procesesing {filename}...')
                relevant = []
                with open(fullname, 'r') as fd:
                    try:
                        doc = fd.readlines()
                        relevant = self.get_relevant_words(30, doc)
                        self.word_cloud(relevant, filename)
                    except (RuntimeError, TypeError, NameError):
                        print(RuntimeError.with_traceback())
                        print(TypeError.with_traceback())
                        print(NameError.with_traceback())
                        print(RuntimeError, TypeError, NameError)

if __name__ == '__main__':
    url = 'http://books.goalkicker.com/'
    pdf_dir = 'pdf/'
    txt_dir = 'txt/'
    img_dir = 'img/'
    ba = Book_Analysis(url, pdf_dir, txt_dir, img_dir)
    #ba.get_urls()
    #ba.to_text()
    ba.word_freq()