from datetime import datetime, timedelta

# Original dictionary
original_dict = {
    "2024-09-08": "ABC hotel",
    "2024-09-09": "ABC hotel",
    "2024-09-10": "sunny hotel",
    "2024-09-11": "sunny hotel",
    "2024-09-12": "sunny hotel",
    "2024-09-13": "ABC hotel",
    "2024-09-14": "ABC hotel",
    "2024-09-15": "sunny hotel",
    "2024-09-16": "rainy hotel"
}

# Initialize the new dictionary
new_dict = {}

# Sort dates in case they are not in order
sorted_dates = sorted(original_dict.keys())

# Iterate through the sorted dates
for i, date in enumerate(sorted_dates):
    hotel = original_dict[date]
    date_obj = datetime.strptime(date, "%Y-%m-%d")

    # If the hotel is not in the new dictionary, initialize it
    if hotel not in new_dict:
        new_dict[hotel] = [{"startDate": date, "endDate": (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")}]
    else:
        # Update the end date if the hotel is the same as the previous day
        prev_date = datetime.strptime(new_dict[hotel][-1]["endDate"], "%Y-%m-%d")
        if prev_date < date_obj:
            new_dict[hotel].append({"startDate": date, "endDate": (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")})
        elif prev_date == date_obj:
            new_dict[hotel][-1]["endDate"] = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

# Print the resulting dictionary
print(new_dict)

# {'ABC hotel': [
#     {'startDate': '2024-09-08', 'endDate': '2024-09-10'}, 
#     {'startDate': '2024-09-13', 'endDate': '2024-09-15'}], 
#  'sunny hotel': [
#      {'startDate': '2024-09-10', 'endDate': '2024-09-13'}, 
#      {'startDate': '2024-09-15', 'endDate': '2024-09-16'}], 
#  'rainy hotel': [
#      {'startDate': '2024-09-16', 'endDate': '2024-09-17'}]
#  }
