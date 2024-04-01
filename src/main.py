import json
import nltk
import wikipediaapi
import os

if not os.path.exists('data.json'):

    # Code permettant de créer data.json, dictionnaire qui map l'id de chaque doc wikipedia 
    # à sa segmentation en passages de 6 phrases

    def data(type): 
        """
        Pour chaquer fichier (dev, test, train), on crée un dictionnaire ayant pr clés question id, doc id...
        type peut être dev, test ou train.
        """
        # Define the path to your text document
        file_path = rf'WikiPassageQA\{type}.txt'
        
        # Initialize an empty dictionary to hold your data
        data_dict = {}
        # articles_with_ids : dict mapping document IDs to their Wikipedia article titles
        articles_with_ids = {}

        # Open the text document for reading
        with open(file_path, 'r') as file:
            # Skip the header line
            next(file)
            # Iterate over each line in the file
            for line in file:
                # Split the line into components based on tabs
                parts = line.strip().split('\t')
                # Extract the individual components
                qid, question, doc_id, doc_name, rel_passages = parts
                # Convert the QID to an integer (if you want it as an integer)
                qid = int(qid)
                # Convert DocumentID to an integer (if needed)
                doc_id = int(doc_id)
                # Split 'RelevantPassages' into a list of integers (if they are always numbers)
                rel_passages = [int(x) for x in rel_passages.split(',')]
                # Populate the dictionary
                data_dict[qid] = {
                    'Question': question,
                    'DocumentID': doc_id,
                    'DocumentName': doc_name,
                    'RelevantPassages': rel_passages
                }
                articles_with_ids[doc_id] = doc_name[:-5]
        
        return data_dict, articles_with_ids

    # Create a Wikipedia object with a specified user agent
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent='WikiPassageQAProject (hatem.mermoz@gmail.com)'
    )

    # Function to get Wikipedia page content
    def get_wiki_content(title):
        page = wiki_wiki.page(title)
        if page.exists():
            return page.text
        else:
            return None

    # Function to segment text into passages of six sentences
    def segment_text(text, sentences_per_passage=6):
        sentences = nltk.sent_tokenize(text)
        passages = [' '.join(sentences[i:i+sentences_per_passage]) for i in range(0, len(sentences), sentences_per_passage)]
        return passages

    # Dictionary to hold article ID and their segmented text
    articles_dict = {}
    full_articles_dict = {} # Obligé de faire en 2 temps pcq sinon l'api pète un câble

    for type in ["train", "dev", "train"]:
        articles_dict = {}
        data_dict, articles_with_ids = data(type)

        for doc_id, title in articles_with_ids.items():
            content = get_wiki_content(title)
            if content:
                content = segment_text(content)
                articles_dict[doc_id] = content 

        # On remplit le "Vrai" dico
                
        for key in articles_dict.keys():
            full_articles_dict[key] = articles_dict[key]

    # On gère les exceptions :
                
    missing_keys = [347, 583, 517, 188, 208, 573, 228, 862]
    titles = ["Encyclopædia_Britannica", "War_in_Afghanistan_(2001–2021)",
            "Baháʼí_Faith","Brussels", "Civil_rights_movement", "São_Paulo", "2007–2008_financial_crisis"
            , "Eastern_Orthodox_Church"]

    for key, title in zip(missing_keys,titles):
        content = get_wiki_content(title)
        full_articles_dict[key] = segment_text(content)

    # Création du fichier
    filename = 'data.json'

    with open(filename, 'w') as f:
        json.dump(full_articles_dict, open(filename, 'w'), indent=4)  



