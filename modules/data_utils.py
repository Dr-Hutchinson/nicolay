# data_utils.py
# Convert JSON to MessagePack (one-time conversion)
import msgpack

with open('data/lincoln_speech_corpus.json', 'r') as file:
    lincoln_data = json.load(file)

with open('data/lincoln_speech_corpus.msgpack', 'wb') as file:
    file.write(msgpack.packb(lincoln_data))

# Convert the loading function to read MessagePack
@st.cache_data(persist="disk")
def load_lincoln_speech_corpus():
    with open('data/lincoln_speech_corpus.msgpack', 'rb') as file:
        return msgpack.unpackb(file.read())

# Similarly, for voyant_word_counts.json
with open('data/voyant_word_counts.json', 'r') as file:
    voyant_data = json.load(file)

with open('data/voyant_word_counts.msgpack', 'wb') as file:
    file.write(msgpack.packb(voyant_data))

@st.cache_data(persist="disk")
def load_voyant_word_counts():
    with open('data/voyant_word_counts.msgpack', 'rb') as file:
        return msgpack.unpackb(file.read())

from concurrent.futures import ThreadPoolExecutor

@st.cache_data(persist="disk")
def load_all_data():
    with ThreadPoolExecutor() as executor:
        lincoln_future = executor.submit(load_lincoln_speech_corpus)
        voyant_future = executor.submit(load_voyant_word_counts)
        index_future = executor.submit(load_lincoln_index_embedded)

        lincoln_data = lincoln_future.result()
        voyant_data = voyant_future.result()
        index_data = index_future.result()

    return lincoln_data, voyant_data, index_data

# Load data in __init__ method
self.lincoln_data, self.voyant_data, self.lincoln_index_df = load_all_data()
