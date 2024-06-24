import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer


class Preprocessor:
    def __init__(self, tokenizer: AutoTokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        nltk.download('stopwords')
        nltk.download('punkt')


    def __call__(self, text: str) -> str:
        text = self.decodeText(text)
        text = self.deleteStopword(text)
        text = self.deleteSpecialCharacters(text)
        text, nTokens = self.deleteTokens(text)

        return text


    def deleteStopword(self, text: str) -> str:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_words = [word for word in word_tokens if word.lower() not in stop_words]
        filtered_sentence = " ".join(filtered_words)

        return filtered_sentence


    def deleteSpecialCharacters(self, text: str) -> str:
        text.replace("``", "")
        text.replace("''", "")
        text.replace("|", "")
        " ".join(text.split())

        return text


    def deleteTokens(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)

        if len(tokens) > self.max_tokens:
            tokens_new = tokens[:self.max_tokens]
            text = self.tokenizer.decode(tokens_new)

        return text, len(tokens)


    # Detect specific encoding and decode text
    def decodeText(self, text: str) -> str:
        text_decoded = text.encode().decode('unicode_escape')

        return text_decoded
