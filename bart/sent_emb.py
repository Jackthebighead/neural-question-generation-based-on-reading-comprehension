from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('bert-base-cased')

# Two lists of sentences
sentences1 = ['When the time of students unwanted sexual attention from an adult?',
             'When did the study say 9.6% of students claim to have received unwanted sexual attention from an adult?',
             'TWhat company designed the "50"?','What "50" is designed by?']

sentences2 = ['What is the time period of this statistic?',
              'What is the time period of this statistic?',
              'Who designs both the "50" as well as the Trophy?','Who designs both the "50" as well as the Trophy?']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))