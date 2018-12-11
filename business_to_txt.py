import pandas as pd

# GLOBAL PARAMETERS
BUSINESS_CSV_FILE = 'yelp_dataset\\csv\\cities\\business_las_vegas.csv'
REVIEW_CSV_FILE = 'yelp_dataset\\csv\\review.csv'


def get_restaurants(raw_business_df):
    """
    Extracts restaurants from business dataframe. Stirps all businesses without categories
    and looks for categories that match 'Restaurants'
    :param raw_business_df: pandas dataframe with all relevant business data
    :return: pandas dataframe with all restaurants
    """
    # strip the entries that don't have anything in the categories subset.
    business_data_clean_categories = business_data.dropna(subset=['categories'])

    # grab all the frames with "Restaurants" category
    restaurant_data_frame = business_data_clean_categories[
        business_data_clean_categories['categories'].str.contains('Restaurants')]

    return restaurant_data_frame


def get_business_id(business_df, business_name):
    """
    Given a business name (string), return the id of the business
    :param business_df: pandas dataframe from yelp
    :param business_name: string for name of the business
    :return: business_id string
    """
    # try to match the name given
    found_business = business_df[business_df['name'] == business_name]
    if found_business.empty:
        print("ERROR, could not find business \""+business_name+"\"")
        exit(-1)

    return found_business['business_id'].iloc[0]


def concat_reviews(reviews):
    """
    Combines all reviews in list into one large string
    :param reviews: list of reviews
    :return: string containing all the reviews
    """
    large_review = ''
    for review in reviews:
        large_review += ' ' + review

    return large_review


def get_all_business_ids(business_df):
    """
    Gets a list of all the business ids
    :param business_df: pandas dataframe from yelp
    :return: list with business_id strings
    """
    return business_df['business_id'].tolist()


def get_reviews(business_id, review_df):
    """
    Gets all the reviews for a given business_id
    :param business_id: string with unique business id
    :param review_df: dataframe containing all reviews
    :return: list containing all the reviews for specified business id
    """
    reviews_df = review_df[review_df['business_id'] == business_id]
    if reviews_df.empty:
        reviews = []
    else:
        reviews = reviews_df['text'].tolist()

    return reviews


def write_reviews_to_file(review, file_num):
    """
    Writes to a file
    :param review: string to be written to the file
    :param file_num: number for the filename
    :return: None
    """
    f = open('test/'+str(file_num)+'.txt', 'w')
    try:
        f.write(review)
    except UnicodeEncodeError:
        print("Had an encoding error. Skipping for now.")
    f.close()


if __name__ == "__main__":

    # load in the file
    business_data = pd.read_csv(BUSINESS_CSV_FILE)

    # grab all the frames with "Restaurants" category
    restaurant_df = get_restaurants(business_data)

    # free up the business data dataframe. Only need to safe restaurants
    del business_data

    # load the reviews into memory. this takes a while
    review_data = pd.read_csv(REVIEW_CSV_FILE)

    # Get all the business IDs
    business_ids = get_all_business_ids(restaurant_df)

    # Initialize file number
    bulk_reviews_number = 0
    for business in business_ids:
        # grab a list of all the reviews
        reviews = get_reviews(business, review_data)
        if not reviews:
            continue

        # combine the reviews into one string
        lump_review = concat_reviews(reviews)

        # write the reviews to a file
        write_reviews_to_file(lump_review, bulk_reviews_number)

        # increment the filename number
        bulk_reviews_number += 1

    print("done")