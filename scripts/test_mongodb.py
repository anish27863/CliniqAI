from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client['cliniqai']

# Create collections
db.create_collection('users')
db.create_collection('predictions')
db.create_collection('gradcam_results')

print("MongoDB collections created successfully!")