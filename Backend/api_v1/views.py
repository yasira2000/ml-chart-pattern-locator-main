from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PatternRequestSerializer, PatternResponseSerializer
import yfinance as yf
import pandas as pd
from datetime import datetime ,timedelta
from utils.patternLocating import locate_patterns
import traceback

import requests 
import os       
import json     


HUGGING_FACE_API_URL = "https://e1khxm12hawdhhji.us-east-1.aws.endpoints.huggingface.cloud" # Example URL, update with yours
HUGGING_FACE_API_TOKEN = os.environ.get("HUGGING_FACE_API_TOKEN")

# Headers for the API request
HF_HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}",
    "Content-Type": "application/json"
}

# Create your views here.

class PatternDetectionView(APIView):
    def get(self, request):
        response = Response()
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "*"
        
        symbol = request.query_params.get('symbol', None)
        start_date_str = request.query_params.get('start_date', None) # Renamed to avoid clash with datetime object
        end_date_str = request.query_params.get('end_date', None)     # Renamed to avoid clash with datetime object
        
        if not all([symbol, start_date_str, end_date_str]):
            return Response(
                {"error": "Missing required parameters: symbol, start_date, end_date"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Fetch data from Yahoo Finance
            # ticker.history's 'start' and 'end' parameters are inclusive
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date_str, end=end_date_str) # Use string dates for yf.Ticker.history
            
            if df.empty:
                return Response(
                    {"error": "No data found for the given symbol and date range"},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # Reset index to make 'Date' a column
            df = df.reset_index()
            
            # Ensure 'Date' is datetime and convert to naive UTC
            df['Date'] = pd.to_datetime(df['Date'])
            if df['Date'].dt.tz is not None:
                df['Date'] = df['Date'].dt.tz_convert('UTC').dt.tz_localize(None)

            # --- Prepare data for Hugging Face Endpoint ---
            # Your handler.py's __call__ method in the HF endpoint expects a list of dictionaries.
            # Make sure the column names here match what your handler.py expects (e.g., 'Open', 'High', 'Low', 'Close', 'Date', 'Volume').
            # ticker.history usually returns these capitalized.
            
            # Format 'Date' column to 'YYYY-MM-DD' string for JSON serialization
            df_for_hf = df.copy()
            if 'Date' in df_for_hf.columns:
                df_for_hf['Date'] = df_for_hf['Date'].dt.strftime('%Y-%m-%d')
            
            # Convert DataFrame to a list of dictionaries
            input_data_for_hf = df_for_hf.to_dict(orient='records')


            # --- START TEMPORARY DEBUGGING CODE IN DJANGO BACKEND ---
            print("\n--- DEBUGGING DATES BEING SENT TO HUGGING FACE ---")
            for i, row_dict in enumerate(input_data_for_hf):
                # We're specifically interested in the 'Date' column
                if 'Date' in row_dict:
                    date_val = row_dict['Date']
                    print(f"Row {i}: Date='{date_val}' (Type: {type(date_val)})")
                    # You can add more checks here if you suspect non-string types
                    if not isinstance(date_val, str):
                        print(f"  WARNING: Date value is not a string!")
                    elif not date_val: # Checks for empty string or None
                        print(f"  WARNING: Date value is empty!")
                    elif len(date_val) != 10: # YYYY-MM-DD has 10 characters
                        print(f"  WARNING: Date string length is not 10!")
                    elif '-' not in date_val:
                        print(f"  WARNING: Date string missing hyphens!")
                else:
                    print(f"Row {i}: 'Date' column is missing from the dictionary!")
            print("--- END DEBUGGING DATES ---")
            # --- END TEMPORARY DEBUGGING CODE ---


            
            # Create the payload expected by Hugging Face Inference Endpoints
            hf_payload = {"inputs": input_data_for_hf}

            print(f"Sending {len(input_data_for_hf)} OHLC data points to Hugging Face Endpoint.")

            # --- Make the request to Hugging Face Endpoint ---
            hf_response = requests.post(
                HUGGING_FACE_API_URL,
                headers=HF_HEADERS,
                json=hf_payload, # requests will automatically serialize this to JSON
                timeout=300 # Increased timeout for potential long inference times
            )

            hf_response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Process the response from Hugging Face ---
            # Assuming your Hugging Face model returns a list of dictionaries (your patterns)
            patterns_dict = hf_response.json()
            print(f"Received {len(patterns_dict)} patterns from Hugging Face Endpoint.")
            
            if not patterns_dict: # Check if the list is empty
                return Response(
                    {"message": "No patterns found by the Hugging Face model for the given data."},
                    status=status.HTTP_200_OK
                )

            # Your original code converts datetime columns in patterns_df to string.
            # If your Hugging Face model already returns them as 'YYYY-MM-DD' strings,
            # you might not need this loop. If it returns them as full ISO strings
            # or epoch, you might need to re-format them here.
            # For simplicity, I'm assuming your HF model output is already in a serializable format.
            # If your model returns patterns_df from the original `locate_patterns`, then this conversion is still needed.
            # Example if patterns_dict items need date re-formatting:
            
            patterns_df = pd.DataFrame(patterns_dict)
            datetime_columns = ['Start', 'End', 'Seg_Start', 'Seg_End', 'Calc_Start', 'Calc_End']
            for col in datetime_columns:
                if col in patterns_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(patterns_df[col]):
                        patterns_df[col] = pd.to_datetime(patterns_df[col]).dt.strftime('%Y-%m-%d')
            patterns_dict = patterns_df.to_dict('records')


            response = Response(patterns_dict, status=status.HTTP_200_OK)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response["Access-Control-Allow-Headers"] = "*"
            return response
            
        except requests.exceptions.RequestException as e:
            error_message = f"Hugging Face API Error: {str(e)}\n"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f"Status Code: {e.response.status_code}\nResponse: {e.response.text}"
            print(error_message)
            traceback.print_exc()
            return Response(
                {"error": f"Failed to get patterns from Hugging Face: {str(e)}", "detail": error_message},
                status=status.HTTP_502_BAD_GATEWAY # 502 Bad Gateway is appropriate for upstream service errors
            )
        except json.JSONDecodeError as e:
            error_message = f"JSON Decode Error from Hugging Face response: {str(e)}\nResponse Text: {hf_response.text if 'hf_response' in locals() else 'No response received'}"
            print(error_message)
            traceback.print_exc()
            return Response(
                {"error": f"Invalid JSON response from Hugging Face: {str(e)}", "detail": error_message},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            error_message = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_message)
            return Response(
                {"error": str(e), "detail": error_message},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Backend/api_v1/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import traceback # For more detailed error logging

# Assuming your locate_patterns function and its dependencies are correctly set up
# in the utils directory. The model is loaded globally within patternLocating.py.
from utils.patternLocating import locate_patterns
# If you later switch to the Gemni version:
# from utils.patternLocatingGemni import locate_patterns


class PatternDetectionAPIView(APIView):
    def get(self, request):
        symbol = request.query_params.get('symbol', None)
        start_date_str = request.query_params.get('start_date', None)
        end_date_str = request.query_params.get('end_date', None)

        if not all([symbol, start_date_str, end_date_str]):
            return Response(
                {"error": "Missing required parameters: symbol, start_date, end_date"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')

            # yfinance end_date is exclusive for daily/weekly/monthly for the range requested.
            # To include the end_date in results, we typically fetch up to end_date + 1 day.
            # For intraday, it's often inclusive.
            # However, for locate_patterns, you want the exact range.
            # The locate_patterns function will process this exact range.
            # For yf.download, to ensure data *up to and including* end_date_str is fetched for daily:
            fetch_end_date_obj = end_date_obj + timedelta(days=1)

            start_d_yf = start_date_obj.strftime('%Y-%m-%d')
            end_d_yf = fetch_end_date_obj.strftime('%Y-%m-%d')

        except ValueError:
            return Response(
                {"error": "Invalid date format. Please use YYYY-MM-DD."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            print(f"DJANGO VIEW (PatternDetection) - Fetching OHLC for locate_patterns: symbol='{symbol}', start='{start_d_yf}', end='{end_d_yf}', interval='1d'")
            # Fetch daily data for pattern location. The locate_patterns function handles windowing.
            ohlc_data_yf = yf.download(symbol, start=start_d_yf, end=end_d_yf, interval='1d')

            if ohlc_data_yf.empty:
                return Response(
                    {"error": f"No OHLC data found from yfinance for symbol {symbol} between {start_date_str} and {end_date_str} (inclusive for pattern analysis)."},
                    status=status.HTTP_404_NOT_FOUND
                )

            # Prepare data for your locate_patterns function
            ohlc_data_for_function = ohlc_data_yf.reset_index()
            # Ensure 'Date' column does not include timezone for consistency if it has one
            if pd.api.types.is_datetime64_any_dtype(ohlc_data_for_function['Date']) and ohlc_data_for_function['Date'].dt.tz is not None:
                ohlc_data_for_function['Date'] = ohlc_data_for_function['Date'].dt.tz_localize(None)


            # Filter the DataFrame to the EXACT user-requested start and end dates AFTER download
            # This is because yf.download might give slightly more data around the edges.
            ohlc_data_for_function['Date'] = pd.to_datetime(ohlc_data_for_function['Date'])
            ohlc_data_for_function = ohlc_data_for_function[
                (ohlc_data_for_function['Date'] >= start_date_obj) &
                (ohlc_data_for_function['Date'] <= end_date_obj)
            ]
            
            if ohlc_data_for_function.empty:
                return Response(
                    {"message": f"No OHLC data within the exact range {start_date_str} to {end_date_str} after fetching for {symbol}."},
                    status=status.HTTP_200_OK
                )


            # Standardize column names that locate_patterns expects (Open, High, Low, Close, Volume)
            # yfinance usually returns them this way for single tickers.
            expected_cols_map = {
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                'Close': 'Close', 'Volume': 'Volume', 'Date': 'Date'
            }
            cols_to_rename = {}
            for col in ohlc_data_for_function.columns:
                if str(col) in expected_cols_map:
                    cols_to_rename[col] = expected_cols_map[str(col)]
            ohlc_data_for_function = ohlc_data_for_function.rename(columns=cols_to_rename)


            # Call your pattern location function
            # Ensure your ohlc_data_for_function has ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            detected_patterns_df = locate_patterns(
                ohlc_data=ohlc_data_for_function,
                use_parallel_processing=False # Recommended for web server context; set to True if you've configured Celery
            )

            if detected_patterns_df is None or detected_patterns_df.empty:
                return Response(
                    {"message": "No patterns located for the given data."},
                    status=status.HTTP_200_OK
                )

            # Convert DataFrame to JSON, ensuring date columns are strings
            date_columns = ['Start', 'End', 'Seg_Start', 'Seg_End', 'Calc_Start', 'Calc_End']
            for col in date_columns:
                if col in detected_patterns_df.columns:
                    # Ensure the column is datetime before trying to format
                    detected_patterns_df[col] = pd.to_datetime(detected_patterns_df[col], errors='coerce')
                    # Format valid dates, leave NaT as None (which becomes null in JSON)
                    detected_patterns_df[col] = detected_patterns_df[col].dt.strftime('%Y-%m-%d')
            
            patterns_json = detected_patterns_df.to_dict(orient='records')
            return Response(patterns_json, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Error in PatternDetectionAPIView: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": f"Error processing request: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class OHLCDataAPIView(APIView):
    def get(self, request):
        response = Response()
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response["Access-Control-Allow-Headers"] = "*"
        
        symbol = request.query_params.get('symbol', None)
        start_date_str = request.query_params.get('start_date', None)
        end_date_str = request.query_params.get('end_date', None)
        period = request.query_params.get('period', None)
        interval = request.query_params.get('interval', '1d')

        if not symbol:
            return Response(
                {"error": "Missing required parameter: symbol"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            df_data = None
            print(f"DJANGO VIEW (OHLCDataAPIView) - Parameters: symbol='{symbol}', start_date='{start_date_str}', end_date='{end_date_str}', period='{period}', interval='{interval}'")

            if start_date_str and end_date_str:
                start_d_obj = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_d_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
                fetch_end_d_str = end_d_obj.strftime('%Y-%m-%d')
                if interval in ["1d", "1wk", "1mo"]:
                    fetch_end_d_str = (end_d_obj + timedelta(days=1)).strftime('%Y-%m-%d')
                start_d_str_formatted = start_d_obj.strftime('%Y-%m-%d')
                print(f"DJANGO VIEW - Fetching with dates: symbol='{symbol}', start='{start_d_str_formatted}', end='{fetch_end_d_str}', interval='{interval}'")
                df_data = yf.download(symbol, start=start_d_str_formatted, end=fetch_end_d_str, interval=interval, progress=False)
            elif period:
                print(f"DJANGO VIEW - Fetching with period: symbol='{symbol}', period='{period}', interval='{interval}'")
                df_data = yf.download(symbol, period=period, interval=interval, progress=False)
            else:
                print(f"DJANGO VIEW - Fetching with default period (1y): symbol='{symbol}', period='1y', interval='{interval}'")
                df_data = yf.download(symbol, period='1y', interval=interval, progress=False)

            print(f"DJANGO VIEW - Raw yfinance data (is empty: {df_data.empty}):")
            if not df_data.empty:
                print(f"Raw columns before modification: {df_data.columns.tolist()}") # Crucial Debug Print
                print(f"Raw index name: {df_data.index.name}")

            if df_data.empty:
                return Response(
                    {"error": f"No data returned by yfinance for symbol {symbol} with the given parameters."},
                    status=status.HTTP_404_NOT_FOUND
                )
            
            # ---- START FIX FOR MULTIINDEX COLUMNS ----
            if isinstance(df_data.columns, pd.MultiIndex):
                # If columns are MultiIndex, e.g., [('Open', 'MSFT'), ('Close', 'MSFT'), ...],
                # we want to use the first level (Open, Close, etc.) as the column names.
                df_data.columns = df_data.columns.droplevel(1) # Drop the ticker level (MSFT)
                print(f"After droplevel(1), columns: {df_data.columns.tolist()}")
            # ---- END FIX FOR MULTIINDEX COLUMNS ----
            
            df_data = df_data.reset_index()
            # print(f"After reset_index, columns: {df_data.columns.tolist()}")

            # Standardize column names to Capitalized (Open, High, Low, Close, Volume, Date)
            rename_map = {}
            for col_name_obj in df_data.columns:
                col_name_str = str(col_name_obj).lower() # Ensure it's a string and lowercased
                if 'open' == col_name_str: rename_map[col_name_obj] = 'Open'
                elif 'high' == col_name_str: rename_map[col_name_obj] = 'High'
                elif 'low' == col_name_str: rename_map[col_name_obj] = 'Low'
                elif 'close' == col_name_str and 'adj' not in col_name_str : rename_map[col_name_obj] = 'Close'
                elif 'adj close' == col_name_str: rename_map[col_name_obj] = 'Adj_close' # Or map to 'Close' if preferred
                elif 'volume' == col_name_str: rename_map[col_name_obj] = 'Volume'
                elif 'date' == col_name_str: rename_map[col_name_obj] = 'Date'
            
            df_data = df_data.rename(columns=rename_map)
            # print(f"After rename attempt, columns: {df_data.columns.tolist()}")
            # print(df_data.head().to_string())

            if 'Date' in df_data.columns:
                df_data['Date'] = pd.to_datetime(df_data['Date'])
                if interval in ["1d", "1wk", "1mo"]:
                     df_data['Date'] = df_data['Date'].dt.strftime('%Y-%m-%d')
                else:
                     df_data['Date'] = df_data['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            columns_to_select = [col for col in required_columns if col in df_data.columns]
            
            if not columns_to_select or 'Date' not in columns_to_select or not any(c in columns_to_select for c in ['Open', 'Close']):
                print(f"DJANGO VIEW - Missing essential columns after processing. Available: {df_data.columns.tolist()}")
                return Response(
                    {"error": "Essential data columns (e.g., Date, Open, Close) are missing after processing."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            ohlc_data_to_send = df_data[columns_to_select]
            
            # print(f"DJANGO VIEW - Data to send head:\n{ohlc_data_to_send.head().to_string()}")
            response = Response(ohlc_data_to_send.to_dict(orient='records'), status=status.HTTP_200_OK)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response["Access-Control-Allow-Headers"] = "*"
            return response

        except ValueError as ve:
            return Response(
                {"error": f"Invalid date format or parameters: {str(ve)}. Please use YYYY-MM-DD for dates."},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            print(f"Error in OHLCDataAPIView for {symbol}: {str(e)}")
            traceback.print_exc()
            return Response(
                {"error": f"Failed to fetch or process OHLC data: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )