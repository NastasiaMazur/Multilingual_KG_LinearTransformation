import urllib.request
import tarfile

url = "https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_inductive.tar.gz"
filename = "wikidata5m_inductive.tar.gz"

# Download the dataset
print("Downloading Wikidata5M inductive split...")
urllib.request.urlretrieve(url, filename)

# Extract the dataset
print("Extracting the archive...")
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()

print("Files extracted:")
print("- wikidata5m_inductive_train.txt")
print("- wikidata5m_inductive_valid.txt")
print("- wikidata5m_inductive_test.txt")
