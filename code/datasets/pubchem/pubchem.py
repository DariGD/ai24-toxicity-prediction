# TODO: create module to collect data from PubChem
#       destination - ???
import requests
import pprint
import json


base_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
base_PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


# function to make HTTP GET call to https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/CCC/CIDs/json,
# get response in JSON format like this:
# {
#   "IdentifierList": {
#     "CID": [
#       6334
#     ]
#   }
# }
# and extract value of the CID property
def smiles_to_cid(smiles):
    response = requests.get(f"{base_PUG_REST}/compound/smiles/{smiles}/CIDs/json")
    return response.json()["IdentifierList"]["CID"][0]


# function to make HTTP GET call to https://pubchem.ncbi.nlm.nih.gov/rest/pug_view//data/compound/{CID}/JSON?heading=Toxicity+Data
def get_toxicity_data(cid):
    response = requests.get(f"{base_PUG_VIEW}/data/compound/{cid}/JSON?heading=Toxicity+Data")
    return response.json()


# function to extract
# .Record.Section[0].Section[0].Section[0].Information[0].Value.StringWithMarkup[0].String
# from JSON response
def get_toxicity_data_value(json_response):
    return json_response["Record"]["Section"][0]["Section"][0]["Section"][0]["Information"][0]["Value"]["StringWithMarkup"][0]["String"]


smiles = input("Smiles: ")
cid = smiles_to_cid(smiles)


print(f"CID: {cid}")
toxicity_data = get_toxicity_data(cid)
json.dump(toxicity_data, open("toxicity.json", "w"), indent=4)
pprint.pp(toxicity_data)
pprint.pp(get_toxicity_data_value(toxicity_data))
