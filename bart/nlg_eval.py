from nlgeval import compute_metrics, compute_individual_metrics
#print('testing on checkpoint:', model_test_checkpoint)

pred_1 = 'When did the study say 9.6% of students claim to have received unwanted sexual attention from an adult?'
pred_2 = 'What were students who experienced a very enthusiastic teacher more likely to do outside of the classroom?'
pred_3 = 'What company designed the "50"?'



#'What "50" is designed by?'
label_1 = ['What is the time period of this statistic?']
label_2 = ['Students exposed to an enthusiastic teacher usually did what more often outside class?']
label_3 = ['Who designs both the "50" as well as the Trophy?']



metrics_dict = compute_individual_metrics(label_3, pred_3)
print(metrics_dict)
print('* Process Finished!')
