import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re, nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from wordcloud import WordCloud
import spacy
nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.colors as mcolors
from collections import Counter
from matplotlib.ticker import FuncFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

data = pd.read_csv('papers.csv')
data.head()

print(data.columns)
print(len(data.columns))

data.shape

data.info()

data.isnull().sum()

sns.set_style("whitegrid")
plt.figure(figsize=(12,6))
graph = sns.catplot(data=data, x='year', kind='count', height=4.5, aspect=2.5, palette='hls')
graph.set_xticklabels(rotation=45, fontsize=14)
graph.set_yticklabels(fontsize=14)
plt.title("Frequence showing number of papers released in different years", size = 20)
plt.savefig("yo.pdf", bbox_inches='tight', format='pdf')

pd.set_option('display.max_colwidth', None)
data['paper_text'].head()

# Word-size histogram before cleaning text

data['Number_of_words'] = data['paper_text'].apply(lambda x: len(str(x).split()))

plt.style.use('ggplot')
plt.figure(figsize=(6, 4))
sns.distplot(data['Number_of_words'], kde=False, color='blue', bins=100)
plt.title("Frequency distribution of number of words for each text extracted", size=15)

plt.figure(figsize=(10, 6))
data['Number_of_words'].plot(kind='box')
plt.title("Word frequence distribution using Box plot", size=15)

data.drop(data[data["Number_of_words"] < 200].index, inplace=True)

data.shape

len(data[(data['Number_of_words'] > 200) & (data['Number_of_words'] < 500)])

plt.style.use('ggplot')
plt.figure(figsize=(6, 4))
sns.distplot(data['Number_of_words'], kde=False, color='cyan', bins=100)
plt.title("Frequency of Number of Words", size=15)
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

#text clean function
def cleaned_text(text):
    clean = re.sub("\n"," ",text)
    clean=clean.lower()
    clean=re.sub(r"[~.,%/:;?_&+*=!-]"," ",clean)
    clean=re.sub("[^a-z]"," ",clean)
    clean=clean.lstrip()
    clean=re.sub("\s{2,}"," ",clean)
    return clean

data['cleaned_paper_text'] = data['paper_text'].apply(cleaned_text)

data['cleaned_paper_text'] = data['cleaned_paper_text'].apply(lambda x: ' ' .join([word for word in x.split() if len(word) > 3]))

type(data['cleaned_paper_text'].head(10))

cloud = WordCloud(colormap='winter', width=600, height=400).generate(str(data['cleaned_paper_text']))
fig = plt.figure(figsize=(13, 18))
plt.axis('off')
plt.imshow(cloud, interpolation='bicubic')

# Text Cleaning (Stopword removal and lemmatization)

stop = stopwords.words('english')
stop.append("also")
data['stop_removed_paper_text'] = data['cleaned_paper_text'].apply(lambda x: ' '. join([word for word in x.split() if word not in stop]))

data["tokenized"] = data['stop_removed_paper_text'].apply(lambda x: nltk.word_tokenize(x))

def word_lemmatizer(text):
  lem_text = [WordNetLemmatizer().lemmatize(i, pos='v') for i in text]
  return lem_text

data['lemmatized'] = data['tokenized'].apply(lambda x: word_lemmatizer(x))
data['lemmatize_joined'] = data['lemmatized'].apply(lambda x: ' '. join(x))

data['Number_of_words_for_cleaned'] = data['lemmatize_joined'].apply(lambda x:len(str(x).split()))

plt.style.use('ggplot')
plt.figure(figsize=(6, 4))
sns.distplot(data['Number_of_words_for_cleaned'],kde = False, color= "navy", bins = 100)
plt.title("Frequency distribution of number of words for each text extracted after removing stopwords and lemmatization", size=16)

data.drop(data[data["Number_of_words_for_cleaned"]>4500].index, inplace = True)

plt.style.use('ggplot')
plt.figure(figsize=(6, 4))
sns.distplot(data['Number_of_words_for_cleaned'],kde = False, color= "orangered", bins = 100)
plt.title("Frequency distribution of no of words in the documents after removing docs containing > 4500 words", size=15)

plt.style.use('classic')
plt.figure(figsize=(14,6))
freq = pd.Series(" ".join(data["lemmatize_joined"]).split()).value_counts()[:30]
freq.plot(kind = "bar", color = "pink")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("30 most frequent words",size=20)
plt.savefig('yo.pdf', bbox_inches='tight', format='pdf')

tokens = data['lemmatize_joined'].apply(lambda x: nltk.word_tokenize(x))

w2v_model = Word2Vec(tokens,
                     min_count=600,
                     window=10,
                     vector_size=250,
                     alpha=0.03,
                     min_alpha=0.0007,
                     workers=4,
                     seed=42)

v1 = w2v_model.wv['model']
v1.size

sim_words = w2v_model.wv.most_similar('estimator')
sim_words

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.index_to_key:
        tokens.append(model.wv[word])
        labels.append(word)

    # Convert tokens list to a NumPy array
    tokens = np.array(tokens)

    tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=2000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(14, 14))

    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(w2v_model)

w2v_model_2 = Word2Vec(tokens,
                     min_count=1000,
                     window=10,
                     vector_size=250,
                     alpha=0.03,
                     min_alpha=0.0007,
                     workers = 4,
                     seed=50)

tsne_plot(w2v_model_2)

# topic modelling using LDA

dictionary = corpora.Dictionary(data["lemmatized"])
doc_term_matrix = [dictionary.doc2bow(rev) for rev in data["lemmatized"]]

doc_term_matrix

LDA = gensim.models.ldamodel.LdaModel

#Build LDA Model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=4, random_state=100, chunksize=200, passes=100)

lda_model.print_topics()

# checking coherence of the topics identified by LDA model

coherence_model_lda = CoherenceModel(model=lda_model, texts=data['lemmatized'], dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

coherence_lda # in the scale of 0 - 1, we got a coherence score of ~0.37, which is not good, it can be improved by hyperparameter tweaking

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)
fig, axes = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=250)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')

def format_topics_sentences(ldamodel=None, corpus=None, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                # sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                sent_topics_df = pd.concat([sent_topics_df, pd.DataFrame([[int(topic_num), round(prop_topic, 4), topic_keywords]])], ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=doc_term_matrix, texts=data["lemmatized"])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data["lemmatized"] for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(10,7), sharey=True, dpi=100)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.show()

import matplotlib.pyplot as plt

# Checking the number of documents that belong to each topic (dominant topic) visually
topic_counts = df_dominant_topic['Dominant_Topic'].value_counts()

# Create a figure and axis
plt.figure(figsize=(12, 6))
topic_counts.plot(kind='bar', color='orange')

# Add titles and labels
plt.title("How many documents belong to each topic", size=14)
plt.xlabel("Topics", size=16)
plt.ylabel("Number of documents", size=16)

# Show the plot
plt.show()