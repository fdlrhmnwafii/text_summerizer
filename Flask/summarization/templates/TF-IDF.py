import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import math

text_string = input()
sentences = sent_tokenize(text_string)

def __create_frequency_table(text_string) -> dict:
    
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()
    
    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

freqTable = __create_frequency_table(text_string)

def _create_frequency_matrix(sentences):
    frequency_matrix ={}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue
                
            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        
        frequency_matrix[sent[:15]] = freq_table
        
    return frequency_matrix

freq_matrix = _create_frequency_matrix(sentences)

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}
    
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        
        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] =count / count_words_in_sentence
            
        tf_matrix[sent] = tf_table
        
    return tf_matrix

tf_matrix = _create_tf_matrix(freq_matrix)

def _create_document_per_words(freq_matrix):
    word_per_doc_table = {}
    
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
                
    return word_per_doc_table

count_doc_per_words = _create_document_per_words(freq_matrix)

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
            
        idf_matrix[sent] = idf_table
        
    return idf_matrix

idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, len(sentences))

def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(),idf_matrix.items()):
        tf_idf_table = {}
        
        for(word1, value1), (word2, value2) in zip(f_table1.items(),
                                                   f_table2.items()):
            tf_idf_table[word1] = float(value1 * value2)
        
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)

def _score_sentences(tf_idf_matrix) -> dict:

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

sentenceValue = _score_sentences(tf_idf_matrix)

def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
        
    average = (sumValues / len(sentenceValue))
    
    return average

threshold = _find_average_score(sentenceValue)

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    
    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
            
    return summary

summary_tfidf = _generate_summary(sentences, sentenceValue, threshold)

print("Hasilnya sebagai Berikut")
print(summary_tfidf)