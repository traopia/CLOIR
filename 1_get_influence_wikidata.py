import requests
from bs4 import BeautifulSoup
from unidecode import unidecode
import pandas as pd
import ast
import numpy as np
from itertools import chain
import json

def normalize_string(s):
    # Remove accents and convert to lowercase
    normalized = unidecode(s.lower())
    # Replace spaces with hyphens
    normalized = normalized.replace(' ', '-')
    return normalized

def normalize_dict(input_dict):
    normalized_dict = {}
    for key, values in input_dict.items():
        # Normalize key
        normalized_key = normalize_string(key)
        # Normalize values
        normalized_values = [normalize_string(value) for value in values]
        normalized_dict[normalized_key] = normalized_values
    return normalized_dict

def filter_dict_by_keys(input_dict, key_list):
    filtered_dict = {key: value for key, value in input_dict.items() if key in key_list}
    return filtered_dict


def get_influenced_by():
    url = "https://query.wikidata.org/sparql"
    query = """
    SELECT ?subject ?subjectLabel ?influencedBy ?influencedByLabel WHERE {
      ?subject wdt:P106/wdt:P279* wd:Q1028181. # Find subjects with occupation or profession as painter or artist
      ?subject wdt:P737 ?influencedBy.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers, params={'query': query})
    data = response.json()
    influenced_by_list = []
    for item in data['results']['bindings']:
        influenced_by_list.append({
            'subject': item['subject']['value'].split('/')[-1],
            'subjectLabel': item['subjectLabel']['value'],
            'influencedBy': item['influencedBy']['value'],
            'influencedByLabel': item['influencedByLabel']['value']
        })
    return influenced_by_list


def clean_dictionary(dictionary):
    cleaned_dict = {}
    for key, values in dictionary.items():
        cleaned_dict[key] = list(set(values))
    return cleaned_dict




def dictionary_influence():
    influence_dict = {}
    influenced_by_list = get_influenced_by()
    for influenced_by in influenced_by_list:
        if influenced_by['subjectLabel'] not in influence_dict:
            influence_dict[influenced_by['subjectLabel']] = []
        influence_dict[influenced_by['subjectLabel']].append(influenced_by['influencedByLabel'])

    influence_dict = normalize_dict(influence_dict)
    influence_dict = clean_dictionary(influence_dict)
    return influence_dict


def scrape_wikiart(artist_url):
    class_names_influenced_by = ['Teachers:', 'Influenced by:']
    influenced_by = []
    class_names_influenced_on = ['Pupils:', 'Influenced on:']
    influenced_on = []
    class_names_similar = ['Friends and Co-workers:']
    friends = []
    # Send a GET request to the URL
    response = requests.get(artist_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all <s> tags in the page
        s_tags = soup.find_all('s')
        for s_tag in s_tags:
            # Find the parent element of the <s> tag
            parent = s_tag.find_parent()
            # Find all <a> tags within the same parent element
            a_tags = parent.find_all('a', {'target': '_self', 'href': True})
            # Extract the text content and relative URLs of all <a> tags found
            targets = [(a_tag.get_text(strip=True), a_tag.get('href')) for a_tag in a_tags]
            # Extract the text content of the <s> tag
            class_name = s_tag.get_text(strip=True)
   
            # Print the class and corresponding targets
            for target_text, href in targets:
                if class_name in class_names_influenced_by:
                    influenced_by.append( href.split('/')[-1])
                elif class_name in class_names_influenced_on:
                    influenced_on.append( href.split('/')[-1])
                elif class_name in class_names_similar:
                    friends.append( href.split('/')[-1])
            influenced_by, influenced_on, friends = list(set(influenced_by)), list(set(influenced_on)), list(set(friends))
        return influenced_by, influenced_on, friends

    else:
        print("Failed to retrieve page:", response.status_code)
        return None, None, None


def scrape_wikiart_influencers(df,min_freq_artist=30):
    artist_freq = df.artist_name.value_counts()
    artist_freq = artist_freq[artist_freq > min_freq_artist]
    artist_freq_more = list(artist_freq.keys())
    influenced_by = {key: '' for key in artist_freq_more}
    influenced_on = {key: '' for key in artist_freq_more}
    friends = {key: '' for key in artist_freq_more}
    for artist in artist_freq_more:
        artist_url = 'https://www.wikiart.org/en/' + artist
        influenced_by[artist], influenced_on[artist], friends[artist] = scrape_wikiart(artist_url)
    return influenced_by, influenced_on, friends



def scrape_wikidata_influencers(df,min_freq_artist=30):
    artist_freq = df.artist_name.value_counts()
    artist_freq = artist_freq[artist_freq > min_freq_artist]
    artist_freq_more = list(artist_freq.keys())
    artists_to_change = ['zinaida-serebryakova','eugene-louis-boudin', 'martiros-saryan','lawrence-alma-tadema','m.-c.-escher', 'maria-prymachenko','joaquin-sorolla']
    artist_freq_more.extend(artists_to_change)
    influence_dict = dictionary_influence()
    painters_influence_dict = filter_dict_by_keys(input_dict=influence_dict, key_list=artist_freq_more)
    keys_to_change = {'zinaida-serebryakova' : 'zinaida-serebriakova' , 'eugene-louis-boudin' : 'eugene-boudin', 
                      'martiros-saryan' : 'martiros-sarian', 'lawrence-alma-tadema' : 'sir-lawrence-alma-tadema', 
                      'm.-c.-escher' : 'm-c-escher', 'maria-prymachenko' : 'maria-primachenko', 'joaquin-sorolla' : 'joaquã\xadn-sorolla'}
    wikidata_influenced = {keys_to_change[old_key] if old_key in keys_to_change else old_key: value for old_key, value in painters_influence_dict.items()}
    return wikidata_influenced


def reverte_dict(original_dict):
    reverted_dict = {}
    for key, values in original_dict.items():
        if values:  # Check if the list of values is not empty
            for value in values:
                if value in reverted_dict:
                    reverted_dict[value].append(key)  # If the key exists, append to the existing list
                else:
                    reverted_dict[value] = [key]  # If the key doesn't exist, create a new list with the key
        else:
            reverted_dict[key] = []  # Handle case where the key's value is an empty list
    return reverted_dict

def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()  # Make a copy of the first dictionary
    for key, value in dict2.items():
        if key in merged_dict:
            if value is not None:
                if merged_dict[key] is not None:
                    merged_dict[key].extend(value)
                else:
                    merged_dict[key] = value  # If the value is None in dict1, replace it with the value from dict2
        else:
            merged_dict[key] = value
    return merged_dict


def main(csv_file_path):
    # influence_dict = dictionary_influence()
    df = pd.read_csv(csv_file_path)

    #consider artists who have at least 100 paintings
    wikiart_influenced_by, wikiart_influenced_on, wikiart_friends = scrape_wikiart_influencers(df,100)
    wikidata_influenced_by = scrape_wikidata_influencers(df,100)
    #combine influend on and influenced by from wikiart
    wikiart_by_on = merge_dictionaries(wikiart_influenced_by,reverte_dict(wikiart_influenced_on))
    
    #combine wikiart and wikidata
    influenced_by_dict = merge_dictionaries(wikiart_by_on, wikidata_influenced_by)

    #filter artists who have at least 30 paintings
    artist_freq_more = df['artist_name'].value_counts()
    artist_freq_more = artist_freq_more[artist_freq_more > 30].index.tolist()
    influenced_by_dict_filtered = filter_dict_by_keys(influenced_by_dict,artist_freq_more)

    #list of influencers artists
    nested_list = list(influenced_by_dict_filtered.values())
    values_list = np.unique(list(chain.from_iterable(sublist for sublist in nested_list)))

    #filter out artists with no influence (not influencers or influenced by anyone)
    filter_out_no_influence = [k for k, v in influenced_by_dict.items() if v == [] and k not in values_list]
    dict_filter_out = {key: value for key, value in influenced_by_dict_filtered.items() if key not in filter_out_no_influence}
    dict_filter_out = clean_dictionary(dict_filter_out)
    with open('DATA/influenced_by_dict.json', 'w') as file:
        json.dump(dict_filter_out, file)
    df['influenced_by'] = df['artist_name'].map(dict_filter_out)
    csv_out_path = csv_file_path.replace('.csv', '_influence.csv')
    df.dropna(subset=['influenced_by'], inplace=True)
    #df.influenced_by = df.influenced_by.apply(lambda x: ', '.join(ast.literal_eval(x)))
    df.influenced_by = df.influenced_by.apply(lambda x: ', '.join(x))
    df.to_csv(csv_out_path, index=False)
    
    return df


if __name__ == '__main__':
    #csv_file_path='DATA/Dataset/wikiart_artists_filtered.csv'
    csv_file_path='DATA/Dataset/wikiart_full.csv'
    main(csv_file_path)