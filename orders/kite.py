import json
import requests


def place_live_order(credential_id, order_details):
    """Function to place live trade order via REST API"""
    try:
        print("Placing Live Trade Order")
        request_dict = {"credential_id": str(credential_id),
                        "order_details": json.dumps(order_details)}
        print("request_dict : {}".format(request_dict))

        response = requests.post(url=orders_url+"place_order", params=request_dict)
        print("******** Order Placement Response *********")
        print(response.json())
        return response.json()
    except Exception as e:
        print("Exception in placing live trade order : {}".format(e))
        return None


def modify_live_order(credential_id, order_details):
    """Function to Modify Live Order"""
    try:
        print("* Modifying Live Order")
        request_dict = {"credential_id": str(credential_id),
                        "order_details": json.dumps(order_details)}
        print("request_dict : {}".format(request_dict))

        response = requests.post(url=orders_url + "modify_order", params=request_dict)
        print("******** Order Placement Response *********")
        print(response.json())
        return response.json()
    except Exception as e:
        print("Exception in Modify Live Order : {}".format(e))
        pass
