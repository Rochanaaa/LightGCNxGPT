import torch
import pandas as pd
from fuzzywuzzy import fuzz
import openai

def make_recommendations(model, edge_index, movie_path):
    """Make movie recommendations

    Args:
        model (LighGCN): lightgcn model
        edge_index (torch.Tensor): 2 by N list of edges
        movie_path (str): path to the movies dataset

    Returns:
        float: ndcg @ k
    """
    df = pd.read_csv(movie_path)
    movie_titles = pd.Series(df.title.values, index=df.movieId).to_dict() #map movies to titles
    movie_genres = pd.Series(df.genres.values, index=df.movieId).to_dict() #map movies to genres

    openai.api_key = " ........."

    #making recommendation process
    recall_model_sum = 0
    precision_model_sum = 0
    precision_final_sum = 0
    recall_final_sum = 0
    precision_total_sum = 0
    recall_total_sum = 0
    recall_rem_sum = 0
    precision_rem_sum = 0
    num_users_with_hr_movies = 0
    accuracy_final_sum = 0
    accuracy_model_sum = 0
    occurences = 0
    doing_better_recall = 0
    doing_better_precision = 0
    user_count_instances = 0
    recall_lightgcn_sum = 0
    precision_lightgcn_sum = 0
    #(rest of the code)

    user_train_pos_items = get_user_positive_items(train_edge_index)
    user_test_pos_items = get_user_positive_items(test_edge_index)
    user_val_pos_items = get_user_positive_items(val_edge_index)


    for user in test_user_indices:
        topK = 20
        # user = user_mapping[user_id] #map the user
        if user in user_pos_items:
            print("===================================")
            print("user", user)
            e_u = model.users_emb.weight[user]
            scores = model.items_emb.weight @ e_u
            #pos_items = user_test_pos_items[user] + user_val_pos_items[user]

            values, indices = torch.topk(scores, k=len(user_pos_items[user]) + topK)

            user_instance = 0

            hrated_movies = [index.cpu().item() for index in indices if index in user_train_pos_items[user]] #movies that were rated highly by the user and model knows
            hrated_movie_ids = [list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)] for movie in hrated_movies] #get the movie ids
            hrated_titles = [movieid_title[id] for id in hrated_movie_ids]
            hrated_genres = [movieid_genres[id] for id in hrated_movie_ids]

            # rec_ids_real = [index.cpu().item() for index in indices] #movies predicted by the user -> raw ids
            rec_ids_real = [index.cpu().item() for index in indices if index not in user_train_pos_items[user]]
            print("movie ids predicted", rec_ids_real)
            topk_movies_rec = rec_ids_real[:topK] #top movies recommended by model
            print("top k movies rec", topk_movies_rec)

            rec_map = [list(movie_mapping.keys())[list(movie_mapping.values()).index(movie)] for movie in topk_movies_rec] #movie ids recommended by model
            rec_titles = [movie_titles[id] for id in rec_map]
            rec_genres = [movie_genres[id] for id in rec_map]
            # gr_user_pos_items = get_user_positive_items(edge_index)
            # ground_truth = gr_user_pos_items[user] #ground truth for the user, all highly rated movies by the user
            ground_truth = user_test_pos_items[user]

            def calculate_recall_precision(gr, recommended_movies):
                # print("ground truth", gr)
                # print("recommended movies", recommended_movies)
                liked_set = set(gr)
                recommended_set = set(recommended_movies)
                true_positives = len(liked_set.intersection(recommended_set))
                false_negatives = len(liked_set.difference(recommended_set))
                false_positives = len(recommended_set.difference(liked_set))
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                total_predictions = len(recommended_movies)
                return recall, precision

            titles_liked_json = json.dumps(hrated_titles)
            titles_rec_json = json.dumps(rec_titles)
            genres_liked_json = json.dumps(hrated_genres)
            print("recommended titles by LightGCN Model", titles_rec_json)
            print("liked movies", titles_liked_json)

            # prompt = "can you recommend movies released before july 2017 that you predict the user will rate highly by profiling them based on the movies they like and other users likes? Base these recommendations on your learnt knowledge of movies similar users are liking and online movies rating sources. return it as a list in this format movies_reco = [" ", " "]. ensure there are quotes and commas."
            # prompt += "this is the movies the user liked"
            # prompt += titles_liked_json
            # prompt += "this is the genres of movies the user liked"
            # prompt += genres_liked_json
            # prompt += "Please remove any movies from the model's predictions that you think does not fit the user profile and preferences, from those towards the end of the list as the recommendations have been ranked by the model already, it is perfoming with an recall of 10%. return it as a list in this format movies_removed = [" ", " "]. ensure there are quotes and commas."
            # prompt += "this is what the model predicted"
            # prompt += titles_rec_json
            prompt = "Can you profile this user and make educated guesses of the following details {age, gender, country, liked genres, disliked genres, liked directors}"
            prompt += "This is the list of movies the user has previously rated really highly"
            prompt += titles_liked_json
            prompt += "Based on their user profile, can you remove any movies from the list of movies predicted by the LightGCN model that you think the model has made a mistake. The movies have been ranked by the model already so focus on removing movies from the ones at the end of the list. return it as a list in this format movies_removed = [" ", " "]. ensure there are quotes and commas."
            prompt += "this is the list of movies the model has recommended for the user"
            prompt += titles_rec_json
            prompt += "can you also recommend any movies released before july 2017 that you think the user will rate highly based on the movies they have previously rated highly. return it as a list in this format movies_reco = [" ", " "]. ensure there are quotes and commas."
            model_choice = "gpt-3.5-turbo"

            def generate_text(prompt, model=model_choice):
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a hardworking assistant."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    movies_removed = response['choices'][0]['message']['content'].strip()
                    return movies_removed
                except Exception as e:
                    return f"An error occurred: {e}"
            generated_text = generate_text(prompt)
            print("GPT Prompt:")
            print(generated_text)
            # print("=======================")

            def extract_movie_lists(generated_text):
                # Define regular expressions to match the lists
                removed_pattern = r"movies_removed\s*=\s*(\[.*?\])"
                reco_pattern = r"movies_reco\s*=\s*(\[.*?\])"

                # Find matches for the patterns in the generated_text
                removed_match = re.search(removed_pattern, generated_text)
                reco_match = re.search(reco_pattern, generated_text)

                try:
                    movies_removed = eval(removed_match.group(1)) if removed_match else None
                except (SyntaxError, NameError):
                    print("Error evaluating movies_removed. Check the formatting.")
                    movies_removed = None

                try:
                    movies_reco = eval(reco_match.group(1)) if reco_match else None
                except (SyntaxError, NameError):
                    print("Error evaluating movies_reco. Check the formatting.")
                    movies_reco = None

                return movies_removed, movies_reco

            movies_removed, movies_reco = extract_movie_lists(generated_text)

            if movies_removed is not None and movies_reco is not None:
                movie_map_rem = [movie_id for movie_id, title in movie_titles.items() if movies_removed is not None and title in movies_removed]

                movie_real_rem = [
            list(movie_mapping.values())[  # Retrieve the keys (movie IDs) of the movie_mapping dictionary
                list(movie_mapping.keys()).index(movie)  # Get the index of the current movie ID in the values of movie_mapping
            ] for movie in movie_map_rem  # Iterate over each movie ID in topk_movies_rec
        ]
                rec_movie_ids_filtered = [movie_id for movie_id in topk_movies_rec if movie_id not in movie_real_rem] #removing movie ids not recommended

                movie_names_reco = [title for title in movies_reco if any(fuzz.partial_ratio(title, movie_title) >= 80 for movie_title in movie_titles.values())]
                print("movie names recommended by GPT", movie_names_reco)
                movie_map_reco = [movie_id for movie_id, title in movie_titles.items() if movie_names_reco is not None and title in movie_names_reco]

                movie_real_reco = [
            list(movie_mapping.values())[list(movie_mapping.keys()).index(movie)] for movie in movie_map_reco]

                recall_model, precision_model = calculate_recall_precision(ground_truth, topk_movies_rec)
                print("========================================================")
                print("Recall of the LightGCN Model: ", recall_model)
                print("Precision of the LightGCN Model: ", precision_model)
                print("========================================================")
                precision_model_sum += precision_model
                recall_model_sum += recall_model
                print("Sum of Recall of LightGCN Model: ", recall_model_sum)
                print("Sum of Precision of LightGCN Model: ", precision_model_sum)

                # rec_movie_ids_total = topk_movies_rec + movie_real_reco
                rec_movie_ids_total = list(set(topk_movies_rec + movie_real_reco)) #ensures only unique ids from both
                rec_movie_ids_final = list(set(rec_movie_ids_filtered + movie_real_reco))
                if len(rec_movie_ids_final) >= topK:
                    rec_movie_ids_final = rec_movie_ids_final[:topK]
                # print("after length of movie ids", len(rec_movie_ids_final))
                else:
                # print("length less than", len(rec_movie_ids_final))
                    remaining = topK - len(rec_movie_ids_final)
                    rec_movie_ids_final = rec_movie_ids_final + movie_real_rem[:remaining]
                    # print("length after adding", len(rec_movie_ids_final))
                    # Initialize counter

                # Iterate through each id in movie_names_reco
                for movie in movie_real_reco:
                    # Check if the id exists in titles_liked_json
                    if movie in hrated_movies:
                    # Check if the id exists in rec_movie_ids_final
                        if movie in rec_movie_ids_final:
                            occurences += 1
                            print("occurences", occurences)
                            user_instance = 1
                if user_instance == 1:
                    user_count_instances += 1
                print("user instances", user_count_instances)

                if len(rec_ids_real) > len(rec_movie_ids_total):
                    cut_off_compare = len(rec_movie_ids_total)
                else:
                    cut_off_compare = len(rec_ids_real)

                recall_lightgcn, precision_lightgcn = calculate_recall_precision(ground_truth, rec_ids_real[:cut_off_compare])
                recall_lightgcn_sum += recall_lightgcn
                precision_lightgcn_sum += precision_lightgcn
                print("Sum of Recall of LightGCN Recommendations: ", recall_lightgcn_sum)
                print("Sum of Precision of LightGCN Recommendations: ", precision_lightgcn_sum)

                print("Sum of Recall after adding GPT Recommendations: ", recall_total_sum)
                print("Sum of Precision after adding GPT Recommendations: ", precision_total_sum)
                recall_total, precision_total = calculate_recall_precision(ground_truth, rec_movie_ids_total[:cut_off_compare])
                precision_total_sum += precision_total
                recall_total_sum += recall_total
                print("Sum of Recall after adding GPT Recommendations: ", recall_total_sum)
                print("Sum of Precision after adding GPT Recommendations: ", precision_total_sum)

                recall_rem, precision_rem = calculate_recall_precision(ground_truth, rec_movie_ids_filtered)
                recall_rem_sum += recall_rem
                precision_rem_sum += precision_rem
                print("Sum of Recall after removing GPT Removal Suggestions: ", recall_rem_sum)
                print("Sum of Precision after removing GPT Removal Suggestions: ", precision_rem_sum)

                recall_final, precision_final = calculate_recall_precision(ground_truth, rec_movie_ids_final)
                precision_final_sum += precision_final
                recall_final_sum += recall_final
                print("========================================================")
                print("Recall of the GPTGCN Model: ", recall_final)
                print("Precision of the GPTGCN Model: ", precision_final)
                print("========================================================")
                print("Sum of Recall of GPTGCN Model: ", recall_final_sum)
                print("Sum of Precision of GPTGCN Model: ", precision_final_sum)
                if recall_final >= recall_model:
                    doing_better_recall += 1
                    print("doing better than lightgcn", doing_better_recall)
                if precision_final >= precision_model:
                    doing_better_precision += 1
                    print("doing better than lightgcn", doing_better_precision)

                num_users_with_hr_movies += 1
                #time.sleep(5)
                print("num of users", num_users_with_hr_movies)
                # print("precision_final:", precision_final_sum)
                # print("recall_final:", recall_final_sum)
                # print("accuracy final:", accuracy_final_sum)
                # print("precision_total:", precision_total_sum)
                # print("recall_total:", recall_total_sum)
                # print("recall rem:", recall_rem_sum)
                # print("precision rem:", precision_rem_sum)
                # print("total_ndcg @", topK, "is", total_ndcg)
                print("==================================================")

        # Calculate averages
        if num_users_with_hr_movies > 0:
            avg_precision_model = precision_model_sum / num_users_with_hr_movies
            avg_recall_model = recall_model_sum / num_users_with_hr_movies
            avg_precision_final = precision_final_sum / num_users_with_hr_movies
            avg_recall_final = recall_final_sum / num_users_with_hr_movies
            avg_recall_total = recall_total_sum / num_users_with_hr_movies
            avg_precision_total = precision_total_sum / num_users_with_hr_movies
            avg_recall_rem = recall_rem_sum / num_users_with_hr_movies
            avg_precision_rem = precision_rem_sum / num_users_with_hr_movies
            # avg_accuracy_final = accuracy_final_sum / num_users_with_hr_movies
            # avg_accuracy_model = accuracy_model_sum / num_users_with_hr_movies
            avg_lightgcn_recall = recall_lightgcn_sum / num_users_with_hr_movies
            avg_lightgcn_precision = precision_lightgcn_sum / num_users_with_hr_movies
            # avg_ndcg = total_ndcg / num_users_with_hr_movies

        else:
            avg_precision_final = 0
            avg_precision_total = 0
            avg_precision_model = 0
            avg_recall_model = 0
            avg_precision_total = 0
            avg_recall_total = 0
            avg_recall_rem = 0
            avg_precision_rem = 0
            # avg_accuracy = 0
            # avg_ndcg = 0
    print("=======================================================")
    print("Num of users counted", num_users_with_hr_movies)
    print("=======================================================")
    print("LightGCN Model")
    print("Average Recall of LightGCN Model: ", avg_recall_model)
    print("Average Precision of LightGCN Model: ", avg_precision_model)
    print("=======================================================")
    print("GPT GCN Model")
    print("Average Recall of GPT GCN Model: ", avg_recall_final)
    print("Average Precision of GPT GCN Model: ", avg_precision_final)
    print("=======================================================")
    print("Other Metrics")
    print("Average Recall including all GPT Recommendations: ", avg_recall_total)
    print("Average Precision including all GPT Recommendations: ", avg_precision_total)
    print("Average Recall removing all GPT Suggestions: ", avg_recall_rem)
    print("Average Precision removing all GPT Suggestions: ", avg_precision_rem)
    # Print the total number of occurrences
    print("Total occurrences in titles_liked_json:", occurences)
    print("Total Number of user instances", user_count_instances)
    print("Number of times GPTGNN is better than LightGCN (recall):", doing_better_recall)
    print("Number of times GPTGNN is better than LightGCN (precision):", doing_better_precision)
