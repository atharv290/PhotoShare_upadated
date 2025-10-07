from flask import Flask, render_template, request, redirect, url_for, jsonify, session,flash
from flask_mysqldb import MySQL
import hashlib
import os
from email.mime.multipart import MIMEMultipart
import base64
import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify
import random
import datetime
import string
import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from MySQLdb.cursors import DictCursor
from sklearn.impute import SimpleImputer
def generate_random_code(length=8):
    characters = string.ascii_letters + string.digits  # Combine letters and digits
    return ''.join(random.choices(characters, k=length))
# Load the dataset
# Load the dataset
if os.path.exists('public_html/Data/photography_dataset.csv'):
    dataset_path = 'public_html/Data/photography_dataset.csv'
    print(f"Dataset found at: {dataset_path}")
    data = pd.read_csv(dataset_path, na_values=["", "NA", "N/A", "null"])  # Handle empty values
    print("Dataset loaded successfully")
else:
    print("Dataset not found. Using default data.")
    data = pd.DataFrame()  # Use an empty DataFrame as a fallback

# Continue with your app logic
if not data.empty:
    # Check if required columns exist
    required_columns = ['Event Type', 'Location', 'Editing Level', 'Additional Services',
                        'Duration (hrs)', 'Photographers', 'Photographer Rating', 'Cost']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}. Skipping model training.")
    else:
        # Define X and y
        X = data.drop(columns=['Cost'])
        y = data['Cost']

        # Preprocessing features
        categorical_features = ['Event Type', 'Location', 'Editing Level', 'Additional Services']
        numeric_features = ['Duration (hrs)', 'Photographers', 'Photographer Rating']

        # Define transformers
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        # Build column transformer
        preprocessor = ColumnTransformer([
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ])

        # Create pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)
        print("Model trained successfully")
else:
    print("No data available for processing. Skipping model training.")
# Example usag
app = Flask(__name__)
# Set the directory for image uploads
app.config['UPLOAD_FOLDER'] = 'public_html/static/uploads'
# Set allowed file extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# Check if the file is an allowed image format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#import mysql.connector
app.secret_key = '1234'
app.config['MYSQL_HOST'] = 'sql12.freesqldatabase.com'
app.config['MYSQL_USER'] = 'sql12801823'
app.config['MYSQL_PASSWORD'] = '6cNzIaiaXD'
app.config['MYSQL_DB'] = 'sql12801823'
app.config['MYSQL_PORT'] = 3306
Dbname = 'sql12801823'
Mysql = MySQL(app)
# Home Page
@app.route('/')
def home():
    if 'email' in session:
        return render_template('HomePage.html')
    return render_template('index.html')  # Main home page
#handle Signup Logic
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        # Hash the password
        password_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        cur = Mysql.connection.cursor()
        try:
            cur.execute(
                "INSERT INTO user_logins (Full_Name, Email, Password) VALUES (%s, %s, %s)",
                (name, email, password_hash)
            )
            Mysql.connection.commit()
            return redirect(url_for('login'))
        except Exception as e:
            return f"Error: {e}", 500
        finally:
            cur.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Hash the entered password for comparison
        hash_p = hashlib.sha256(password.encode("utf-8")).hexdigest()
        cur = Mysql.connection.cursor()
        try:
            cur.execute("SELECT Email, Password FROM user_logins WHERE Email = %s", (email,))
            user = cur.fetchone()
            if user and hash_p == user[1]:  # Compare hashed passwords
                session['email'] = user[0]  # Store email in session
                return redirect(url_for('home'))
            else:
                return render_template('index.html', error="Invalid email or password")
        except Exception as e:
            return f"Error: {e}", 500
        finally:
            cur.close()
    return render_template('index.html')
#handle logout Logic
@app.route('/logout')
def logout():
    session.pop('admin_email',None)
    session.pop('email',None)
    return redirect(url_for('home'))

#handle Admin Signup Logic
@app.route('/admin_s',methods= ['GET','POST'])
def admin_s():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        cur = Mysql.connection.cursor()
        cur.execute(
    "INSERT INTO admin_logins (Full_Name, Email, Password) VALUES (%s, %s, %s)",
    (name, email, password))
        Mysql.connection.commit()
        cur.close()
        return render_template('admin.html')
    return render_template('admin.html')

#handle Admin login Logic
@app.route('/admin_l', methods=['GET', 'POST'])
def admin_l():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cur = Mysql.connection.cursor()
        cur.execute("SELECT Email, Password FROM admin_logins WHERE Email = %s", (email,))
        user = cur.fetchone()
        cur.execute("SELECT Full_Name, bio, profile_img FROM admin_logins WHERE Email = %s", (email,))
        user1 = cur.fetchone()
        cur.close()
        if user1:
            profile_data = {
                'name': user1[0], 
                'bio': user1[1],   
                'profile_img': "uploads/" + user1[2] if user1[2] else "static/uploads/default-profile.jpg"
            }
        else:
            profile_data = None  
        if user and password == user[1]:  
            session['admin_email'] = user[0]
            return render_template('adminHome.html', profile=profile_data)
        else:
            return render_template('admin.html')
    return render_template('admin.html')

#loads Admin Signup and login page 
@app.route('/admin')
def admin_loging():
    return render_template('admin.html')

#handle Admin Signup Logic
@app.route('/upload_photos', methods=['POST'])
def upload_files():
    files = request.files.getlist('photos')
    cost = request.form['cost']
    if not files:
        return "No files uploaded!", 400

    # Create the 'uploads' directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    random_code = generate_random_code()
    status = request.form['status']

    try:
        cur = Mysql.connection.cursor()
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)  # Save the file to the server's folder
                date = datetime.datetime.now()  # Current datetime
                owner = session['admin_email']

                if status == 'private':
                    email = request.form['clientEmail']
                    name = request.form['clientName']
                    mobile = request.form['clientMobile']
                    query = "INSERT INTO uploads_details (client_name, client_email, client_no, date, owner) VALUES (%s, %s, %s, %s, %s)"
                    cur.execute(query, (name, email, mobile, date, owner))
                    query = "INSERT INTO images2 (image_path, filename, Status, code, client_email, Date, client_name, client_no, price, owner) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    cur.execute(query, (file_path, filename, status, random_code, email, date, name, mobile, cost, owner))
                else:
                    query = "INSERT INTO images2 (image_path, filename, Status, Date, price) VALUES (%s, %s, %s, %s, %s)"
                    cur.execute(query, (file_path, filename, status, date, cost))

        Mysql.connection.commit()
        return "Files uploaded and paths stored in the database successfully!", 200

    except Exception as err:
        return f"Error: {err}", 500
    finally:
        cur.close()
@app.route('/marketplace', methods=['POST'])
def loadpage():
    access_code = request.form['secretCode']
    access_type = request.form['accessType']
    session['secretCode'] = access_code
    session['accessType'] = access_type
    return render_template('marketplace.html')
# Image Serving Route
@app.route('/get-images', methods=['GET'])
def get_images():
    try:
        # Get page and per_page values from request (default values are set)
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 12))  # Show 12 images per page by default

        cur = Mysql.connection.cursor()

        # Modify the query to support LIMIT and OFFSET for pagination
        if session.get('accessType') == 'private':
            query = "SELECT filename,id, image_path, price FROM images2 WHERE code = %s LIMIT %s OFFSET %s"
            cur.execute(query, (session['secretCode'], per_page, (page - 1) * per_page))
        else:
            query = "SELECT filename,id,image_path, price FROM images2 WHERE Status = %s LIMIT %s OFFSET %s"
            cur.execute(query, ('public', per_page, (page - 1) * per_page))

        rows = cur.fetchall()

        if not rows:
            return jsonify({'message': 'No images found'}), 404

        images = []
        for row in rows:
            filename,id, image_path, price = row  # Ensure correct unpacking
            image_url = f"static/uploads/{filename}"  # Construct the URL for accessing the image
            
            images.append({
                'url': image_url,
                'filename': filename,
                'price': price,
                'image_id':id # Include price in the response
            })

        # You can also return the total number of pages or images to help with frontend pagination
        cur.execute("SELECT COUNT(*) FROM images2 WHERE Status = %s", ('public',))  # Total count query for pagination
        total_images = cur.fetchone()[0]
        total_pages = (total_images // per_page) + (1 if total_images % per_page > 0 else 0)

        return jsonify({
            'images': images,
            'total_pages': total_pages,
            'current_page': page
        }), 200

    except Exception as err:
        print(f"Error occurred in get_images: {err}")  # Log the error
        return jsonify({'error': str(err)}), 500

    finally:
        if 'cur' in locals() and cur:
            cur.close()

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'admin_email' not in session:
        return "Unauthorized", 403  

    new_name = request.form.get('name')
    new_bio = request.form.get('bio')
    file = request.files.get('profile_pic')

    try:
        cur = Mysql.connection.cursor()
        email = session['admin_email']
        
        # Fetch old profile image path
        cur.execute("SELECT profile_img FROM admin_logins WHERE Email = %s", (email,))
        row = cur.fetchone()
        old_profile_img = row[0] if row and row[0] else None

        # Initialize image_path variable
        image_path = None

        if file and file.filename != '' and allowed_file(file.filename):
            # Use the same UPLOAD_FOLDER as other functions
            upload_dir = app.config['UPLOAD_FOLDER']
            
            # Ensure upload directory exists
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
                print(f"Created directory: {upload_dir}")

            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_dir, filename)

            # Save the new file
            file.save(file_path)
            print(f"File saved to: {file_path}")

            # Delete old file if it exists
            if old_profile_img:
                # Extract filename from old path and build full path
                old_filename = os.path.basename(old_profile_img)
                old_file_path = os.path.join(upload_dir, old_filename)
                
                if os.path.exists(old_file_path) and old_filename != filename:
                    try:
                        os.remove(old_file_path)
                        print(f"Deleted old file: {old_file_path}")
                    except Exception as e:
                        print(f"Could not delete old image: {e}")

            # Store only the filename in database (same as get_images_photographer)
            image_path = filename

        else:
            # If no new file, keep the existing filename
            if old_profile_img:
                image_path = os.path.basename(old_profile_img)

        # Update database with new info - store only filename
        cur.execute(
            "UPDATE admin_logins SET Full_Name = %s, bio = %s, profile_img = %s WHERE Email = %s",
            (new_name, new_bio, image_path, email)
        )
        Mysql.connection.commit()

        # Fetch updated data for rendering
        cur.execute("SELECT Full_Name, bio, profile_img FROM admin_logins WHERE Email = %s", (email,))
        user = cur.fetchone()

        # Construct URL exactly like get_images_photographer does
        if user[2]:  # If profile_img exists
            profile_img_url = f"uploads/{user[2]}"
        else:
            profile_img_url = "static/uploads/default-profile.jpg"

        profile_data = {
            'name': user[0],
            'bio': user[1],
            'profile_img': profile_img_url
        }

        return render_template('adminHome.html', profile=profile_data)

    except Exception as e:
        print(f"Error in update_profile: {e}")
        return f"Error: {e}", 500

    finally:
        cur.close()

@app.route('/get-images-photographer', methods=['GET'])
def get_images_photographer():
    try:
        cur = Mysql.connection.cursor()
        #caption	Email	Likes	post
        query = "SELECT caption,post,Likes,filename FROM photographer_posts WHERE Email = %s"
        cur.execute(query, (session['admin_email'],))  # Assuming 'public' status for non-private images
        rows = cur.fetchall()
        if not rows:
            return jsonify({'message': 'No images found'}), 404
        
        images = []
        for row in rows:
            caption, post, likes,filename = row  # Ensure correct unpacking
            image_url = f"static/uploads/{filename}" # Construct the URL for accessing the image
            images.append({
                'url': image_url,
                'caption':caption,
                'likes': likes  # Include price in the response
            })

        return jsonify({'images': images}), 200
    
    except Exception as err:
        print(f"Error occurred in get_images: {err}")  # Log the error
        return jsonify({'error': str(err)}), 500
    
    finally:
        if 'cur' in locals() and cur:
            cur.close()
@app.route('/post_img', methods=['POST'])
def post_images():
    files = request.files.getlist('Post_imgs')
    Caption = request.form['caption']
    email = session['admin_email']
    if not files:
        return "No files uploaded!", 400
    # Create the 'uploads' directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    print(Caption)
    try:
        cur = Mysql.connection.cursor()
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)  # Save the file to the server's folder
                date = datetime.datetime.now()  # Current datetime
                # Insert the file path into the database
                #	caption	Email	Likes	post
                query = "INSERT INTO photographer_posts (caption, post, Email, filename) VALUES (%s, %s, %s,%s)"
                cur.execute(query, (Caption, file_path, email,filename))
            Mysql.connection.commit()
            return "Files uploaded and paths stored in the database successfully!", 200
    except Exception as err:
        return f"Error: {err}", 500
    finally:
        cur.close()
@app.route('/save_image_view', methods=['POST'])
def save_image_view():
   try:
        data = request.get_json()  # Get the JSON data from the request
        if not data:
            raise ValueError("No data received")
        print(session['email'])
        email = session['email']
        image_name = data.get('image_name')
        date = datetime.datetime.now() 
        # Save to database
        cur = Mysql.connection.cursor()
      # Corrected query
        query = "INSERT INTO views (client_email, image_url, Date) VALUES (%s, %s, %s)"
        # Assuming 'email', 'image_name', and 'date' are already defined
        cur.execute(query, (email, image_name, date))
        Mysql.connection.commit()
        return jsonify({'status': 'success', 'message': 'Data saved successfully'})
   except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
   
@app.route('/fetchProfiles', methods=['GET'])
def fetch_profiles():
    try:
        cur = Mysql.connection.cursor()
        query = "SELECT Full_name, profile_img, bio,Email, ratings FROM admin_logins"
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            return jsonify({'profile': []}), 200
        #event_type, duration, photographers, location, editing_level, additional_services, photographer_rating
        query2 = """SELECT event_type, duration, photographers_count, event_location, 
                   editing_level, additional_services, photographer_rating 
            FROM photographer_hiring_requests 
            WHERE email = %s 
            ORDER BY created_at DESC 
            LIMIT 1"""
        cur.execute(query2, (session['email'],))  # Execute the query
        latest_entry = cur.fetchone()  # Fetch the latest row
        if latest_entry:
            event_type, duration, photographers_count, event_location, editing_level, additional_services, photographer_rating = latest_entry
            print("Email:", session['email'])
            print("Event Type:", event_type)
            print("Duration:", duration)
            print("Photographers Count:", photographers_count)
            print("Event Location:", event_location)
            print("Editing Level:", editing_level)
            print("Additional Services:", additional_services)
            print("Photographer Rating:", photographer_rating)
        else:
            print("No records found for this email.")
        profiles = []
        for row in rows:
            full_name, profile_img, bio ,email,ratings= row  
            image_url = "/static/uploads/"+profile_img if profile_img else "/static/uploads/default-profile.jpg" 
            cost = estimate_cost(event_type, duration, photographers_count,event_location, editing_level, additional_services, ratings)
            cost = cost+4000
            profiles.append({
                'url': image_url,
                'Full_name': full_name,
                'bio': bio,
                'email':email,
                'cost' : cost
            })

        return jsonify({'profile': profiles}), 200
    
    except Exception as err:
        print(f"Error occurred in fetch_profiles: {err}")  
        return jsonify({'error': str(err)}), 500
    
    finally:
        if 'cur' in locals() and cur:
            cur.close()

@app.route('/photographers_profiles')
def photographers_profiles():
    return render_template('photographers_profiles.html')

@app.route('/photographer/<email>')
def photographer_profile(email):
    cur = Mysql.connection.cursor()

    # Fetch photographer details
    query = "SELECT Full_Name, profile_img, bio FROM admin_logins WHERE Email = %s"
    cur.execute(query, (email,))
    photographer = cur.fetchone()

    if not photographer:
        return "Photographer not found", 404

    full_name, profile_img, bio = photographer
    profile_img ="uploads/"+ profile_img if profile_img else "static/uploads/default-profile.jpg"

    # Fetch photographer's images
    query_images = "SELECT filename FROM photographer_posts WHERE Email = %s"
    cur.execute(query_images, (email,))
    images = cur.fetchall()
    cur.close()
    photographer_images = [{"url": url_for('static', filename=f"uploads/{row[0]}")} for row in images]
    return render_template('adminProfile.html', 
                           name=full_name, 
                           profile_img=profile_img, 
                           bio=bio, 
                           images=photographer_images,
                           photographer_email=email)
@app.route('/save-details', methods=['POST'])
def save_details():
    try:
        # Check if JSON or Form Data is received
        data = request.form if request.form else request.get_json()

        # Extract fields
        name = data.get('name')
        email = session['email']
        event_type = data.get('event_type')
        duration = data.get('duration')
        num_photographers = data.get('num_photographers')
        location = data.get('location')
        editing_level = data.get('editing_level')
        additional_services = data.get('additional_services', '')
        photographer_rating = 4
        additional_details = data.get('additional_details', '')
        # Validate required fields
        if not all([name, email, event_type, duration, num_photographers, location, editing_level, photographer_rating]):
            return jsonify({"error": "All required fields must be filled!"}), 400

        # Insert into database
        cur = Mysql.connection.cursor()
        sql = """INSERT INTO photographer_hiring_requests 
                 (name, email, event_type, duration, photographers_count, event_location, editing_level, additional_services, photographer_rating, additional_details) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (name, email, event_type, duration, num_photographers, location, editing_level, additional_services, photographer_rating, additional_details)
        cur.execute(sql, values)
        Mysql.connection.commit()
        cur.close()
        print("Cost : ",estimate_cost(event_type,duration,num_photographers,location,editing_level,additional_services,photographer_rating))
        return render_template("photographers_Profiles.html")
        #return jsonify({"message": "Hire request submitted successfully!"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def estimate_cost(event_type, duration, photographers, location, editing_level, additional_services, photographer_rating):
    input_data = pd.DataFrame([[event_type, duration, photographers, location, editing_level, additional_services, photographer_rating]],
                              columns=['Event Type', 'Duration (hrs)', 'Photographers', 'Location', 'Editing Level', 'Additional Services', 'Photographer Rating'])
    return model.predict(input_data)[0]

def get_posts():

    cursor = Mysql.connection.cursor()
    cursor.execute("SELECT filename,caption,Like_count,Email FROM photographer_posts")  # Modify based on your DB schema

    posts = [{"filename": row[0], "image_url": f"static/uploads/{row[0]}", 
          "caption": row[1], "Like_count": row[2], "Email": row[3]} 
         for row in cursor.fetchall()]

    cursor.close()
    return posts

@app.route('/get-posts')
def fetch_posts():
    posts = get_posts()
    return jsonify(posts)

@app.route('/like', methods=['POST'])
def like_post():
    try:
        post_id = request.json.get('post_id')   # Email (user identifier)
        filename = request.json.get('filename')  # Filename of the post

        if not post_id or not filename:
            return jsonify({'error': 'Missing post_id or filename'}), 400

        cursor = Mysql.connection.cursor()

        # Check if the user already liked the post
        cursor.execute("SELECT * FROM likes WHERE email = %s AND filename = %s", (post_id, filename))
        like_entry = cursor.fetchone()

        if like_entry:
            # Unlike: Remove from the likes table
            cursor.execute("DELETE FROM likes WHERE email = %s AND filename = %s", (post_id, filename))
            cursor.execute("UPDATE photographer_posts SET like_count = like_count - 1 WHERE filename = %s", (filename,))
            Mysql.connection.commit()
            liked = False  # Now unliked
        else:
            # Like: Add to the likes table
            cursor.execute("INSERT INTO likes (email, filename) VALUES (%s, %s)", (post_id, filename))
            cursor.execute("UPDATE photographer_posts SET like_count = like_count + 1 WHERE filename = %s", (filename,))
            Mysql.connection.commit()
            liked = True  # Now liked

        # Get the updated like count
        cursor.execute("SELECT like_count FROM photographer_posts WHERE filename = %s", (filename,))
        updated_post = cursor.fetchone()
        
        cursor.close()

        if updated_post:
            return jsonify({'like_count': updated_post[0], 'liked': liked})
        else:
            return jsonify({'error': 'Post not found'}), 404

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
@app.route('/hire', methods=['POST'])
def hire():
    try:
        # Check if user is logged in
        if 'email' not in session:
            return jsonify({'success': False, 'message': 'Please login first'}), 401
        
        data = request.form
        print("data received:", data)
        print("user email:", session['email'])
        
        user_email = session['email']
        photographer_email = data.get('photographerEmail')
        event_type = data.get('eventType')
        event_date = data.get('eventDate')
        event_location = data.get('eventLocation')
        contact_info = data.get('contactInfo')
        special_requests = data.get('specialRequests')
        additional_info = data.get('additionalInfo')
        budget = data.get('budget')
        
        # Validate required fields
        if not all([user_email, photographer_email, event_type, event_date, event_location, contact_info, budget]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # Insert data into the MySQL database
        cursor = Mysql.connection.cursor()  # Fixed: should be connection, not connect
        query = """INSERT INTO hires 
                   (user_email, photographer_email, event_type, event_date, event_location, 
                    contact_info, special_requests, additional_info, budget, status) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')"""
        values = (user_email, photographer_email, event_type, event_date, event_location, 
                 contact_info, special_requests, additional_info, budget)
        
        cursor.execute(query, values)
        Mysql.connection.commit()  # Added: commit the transaction
        cursor.close()
        
        return jsonify({'success': True, 'message': 'Hiring request sent successfully!'})
        
    except Exception as e:
        print(f"Error in hire function: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500

@app.route('/get_private_codes', methods=['GET'])
def get_private_codes():
    try:
        email = session['email']
        print(email)
        cur = Mysql.connection.cursor()
        query = """
            SELECT code, client_email, COUNT(*) AS occurrences 
            FROM images2 
            WHERE code IS NOT NULL AND client_email = %s 
            GROUP BY code, client_email 
            HAVING COUNT(*) > 1;
        """
        cur.execute(query, (email,))
        rows = cur.fetchall()
        codes = [row[0] for row in rows]
        return jsonify({'codes': codes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cur.close()
@app.route('/get-past-events', methods=['GET'])
def get_past_events():
    if 'admin_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    email = session['admin_email']
    print(email)
    try:
        cur = Mysql.connection.cursor()
        cur.execute("""
            SELECT client_name, client_email, client_no, date
            FROM  uploads_details
            WHERE owner = %s
            GROUP BY client_email, DATE(date)
            ORDER BY date DESC
        """, (email,))
        data = cur.fetchall()
        cur.close()
        
        results = [{
            'client_name': row[0],
            'client_email': row[1],
            'client_no': row[2],
            'date': str(row[3]),
        } for row in data]

        return jsonify({'clients': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this endpoint to get hiring requests for the photographer
@app.route('/get-hiring-requests', methods=['GET'])
def get_hiring_requests():
    if 'admin_email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    photographer_email = session['admin_email']
    
    try:
        cur = Mysql.connection.cursor()
        # Query to get hiring requests for this photographer
        query = """
            SELECT id, user_email, photographer_email, event_type, event_date, 
                   event_location, contact_info, special_requests, additional_info, 
                   budget, status, created_at
            FROM hires 
            WHERE photographer_email = %s 
            ORDER BY created_at DESC
        """
        cur.execute(query, (photographer_email,))
        rows = cur.fetchall()
        
        requests = []
        for row in rows:
            requests.append({
                'id': row[0],
                'user_email': row[1],
                'photographer_email': row[2],
                'event_type': row[3],
                'event_date': str(row[4]) if row[4] else None,
                'event_location': row[5],
                'contact_info': row[6],
                'special_requests': row[7],
                'additional_info': row[8],
                'budget': row[9],
                'status': row[10],
                'created_at': str(row[11]) if row[11] else None
            })
        
        return jsonify({'requests': requests}), 200
        
    except Exception as e:
        print(f"Error fetching hiring requests: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if 'cur' in locals() and cur:
            cur.close()

# Add this endpoint to handle accept/decline responses
@app.route('/respond-to-request', methods=['POST'])
def respond_to_request():
    if 'admin_email' not in session:
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        request_id = data.get('request_id')
        response = data.get('response')  # 'accepted' or 'declined'
        
        if not request_id or response not in ['accepted', 'declined']:
            return jsonify({'success': False, 'message': 'Invalid request data'}), 400
        
        cur = Mysql.connection.cursor()
        
        # First verify the request belongs to this photographer
        cur.execute("SELECT photographer_email FROM hires WHERE id = %s", (request_id,))
        request_data = cur.fetchone()
        
        if not request_data:
            return jsonify({'success': False, 'message': 'Request not found'}), 404
            
        if request_data[0] != session['admin_email']:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 403
        
        # Update the request status
        update_query = "UPDATE hires SET status = %s WHERE id = %s"
        cur.execute(update_query, (response, request_id))
        Mysql.connection.commit()
        
        # If accepted, you might want to send an email notification here
        if response == 'accepted':
            # Get user email to send confirmation
            cur.execute("SELECT user_email FROM hires WHERE id = %s", (request_id,))
            user_email = cur.fetchone()[0]
            print(f"Sending confirmation to {user_email}")  # Replace with actual email sending
        
        return jsonify({'success': True, 'message': f'Request {response} successfully'})
        
    except Exception as e:
        print(f"Error responding to request: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        if 'cur' in locals() and cur:
            cur.close()

@app.route('/get-user-notifications', methods=['GET'])
def get_user_notifications():
    if 'email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    user_email = session['email']
    
    try:
        cur = Mysql.connection.cursor()
        # Query to get hiring requests for this user
        query = """
            SELECT id, user_email, photographer_email, event_type, event_date, 
                   event_location, contact_info, special_requests, additional_info, 
                   budget, status, created_at
            FROM hires 
            WHERE user_email = %s 
            ORDER BY created_at DESC
        """
        cur.execute(query, (user_email,))
        rows = cur.fetchall()
        
        requests = []
        for row in rows:
            requests.append({
                'id': row[0],
                'user_email': row[1],
                'photographer_email': row[2],
                'event_type': row[3],
                'event_date': str(row[4]) if row[4] else None,
                'event_location': row[5],
                'contact_info': row[6],
                'special_requests': row[7],
                'additional_info': row[8],
                'budget': row[9],
                'status': row[10],
                'created_at': str(row[11]) if row[11] else None
            })
        
        return jsonify({'requests': requests}), 200
        
    except Exception as e:
        print(f"Error fetching user notifications: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if 'cur' in locals() and cur:
            cur.close()


# Create cart table first
def create_cart_table():
    try:
        cur = Mysql.connection.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS cart (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255) NOT NULL,
                image_id INT NOT NULL,
                filename VARCHAR(255) NOT NULL,
                price DECIMAL(10, 2) NOT NULL,
                image_url TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_email) REFERENCES user_logins(Email),
                FOREIGN KEY (image_id) REFERENCES images2(id)
            )
        """)
        Mysql.connection.commit()
        cur.close()
        print("Cart table created successfully")
    except Exception as e:
        print(f"Error creating cart table: {e}")

# Call this when your app starts
create_cart_table()

@app.route('/add-to-cart', methods=['POST'])
def add_to_cart():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        data = request.get_json()
        user_email = session['email']
        image_id = data.get('image_id')
        filename = data.get('filename')
        price = data.get('price')
        image_url = data.get('image_url')
        
        # Check if item already in cart
        cur = Mysql.connection.cursor()
        cur.execute("SELECT id FROM cart WHERE user_email = %s AND image_id = %s", (user_email, image_id))
        existing_item = cur.fetchone()
        
        if existing_item:
            return jsonify({'success': False, 'message': 'Item already in cart'}), 400
        
        # Add to cart
        cur.execute("""
            INSERT INTO cart (user_email, image_id, filename, price, image_url) 
            VALUES (%s, %s, %s, %s, %s)
        """, (user_email, image_id, filename, price, image_url))
        
        Mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'Item added to cart'})
        
    except Exception as e:
        print(f"Error adding to cart: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get-cart-items', methods=['GET'])
def get_cart_items():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        user_email = session['email']
        cur = Mysql.connection.cursor()
        cur.execute("""
            SELECT id, image_id, filename, price, image_url 
            FROM cart 
            WHERE user_email = %s 
            ORDER BY added_at DESC
        """, (user_email,))
        
        items = []
        for row in cur.fetchall():
            items.append({
                'cart_id': row[0],
                'image_id': row[1],
                'filename': row[2],
                'price': float(row[3]),
                'image_url': row[4]
            })
        
        cur.close()
        return jsonify({'success': True, 'items': items})
        
    except Exception as e:
        print(f"Error getting cart items: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get-cart-count', methods=['GET'])
def get_cart_count():
    if 'email' not in session:
        return jsonify({'count': 0})
    
    try:
        user_email = session['email']
        cur = Mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM cart WHERE user_email = %s", (user_email,))
        count = cur.fetchone()[0]
        cur.close()
        return jsonify({'count': count})
    except Exception as e:
        print(f"Error getting cart count: {e}")
        return jsonify({'count': 0})

@app.route('/remove-from-cart', methods=['POST'])
def remove_from_cart():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        data = request.get_json()
        cart_id = data.get('cart_id')
        user_email = session['email']
        
        cur = Mysql.connection.cursor()
        cur.execute("DELETE FROM cart WHERE id = %s AND user_email = %s", (cart_id, user_email))
        Mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'Item removed from cart'})
        
    except Exception as e:
        print(f"Error removing from cart: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/checkout', methods=['POST'])
def checkout():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        user_email = session['email']
        cur = Mysql.connection.cursor()
        
        # Get cart items with filenames
        cur.execute("SELECT image_id, filename, price FROM cart WHERE user_email = %s", (user_email,))
        cart_items = cur.fetchall()
        
        if not cart_items:
            return jsonify({'success': False, 'message': 'Cart is empty'}), 400
        
        # Calculate total amount
        total_amount = sum(float(item[2]) for item in cart_items)
        
        # Store filenames in session for download
        filenames = [item[1] for item in cart_items]
        
        # Clear the cart
        cur.execute("DELETE FROM cart WHERE user_email = %s", (user_email,))
        Mysql.connection.commit()
        cur.close()
        
        # If free items, store filenames in session for download
        if total_amount == 0:
            session['download_filenames'] = filenames
            return jsonify({
                'success': True, 
                'message': f'Order placed successfully! Total: ₹{total_amount:.2f}',
                'total': total_amount,
                'download_required': True
            })
        
        return jsonify({
            'success': True, 
            'message': f'Order placed successfully! Total: ₹{total_amount:.2f}',
            'total': total_amount,
            'download_required': False
        })
        
    except Exception as e:
        print(f"Error during checkout: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/download-free-images')
def download_free_images():
    if 'email' not in session or 'download_filenames' not in session:
        return "No download available", 404
    
    try:
        filenames = session.get('download_filenames', [])
        
        if not filenames:
            return "No images to download", 400
        
        # Create a zip file containing all images
        import zipfile
        import os
        from io import BytesIO
        
        zip_buffer = BytesIO()
        successful_downloads = 0
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, filename in enumerate(filenames):
                try:
                    # Build the full file path
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    # Check if file exists
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as img_file:
                            # Use original filename with sequence number for organization
                            zip_filename = f"{i+1:02d}_{filename}"
                            zip_file.writestr(zip_filename, img_file.read())
                        successful_downloads += 1
                        print(f"Successfully added {filename} to zip")
                    else:
                        print(f"File not found: {file_path}")
                except Exception as e:
                    print(f"Error adding {filename} to zip: {e}")
                    continue
        
        zip_buffer.seek(0)
        
        # Clear download session
        session.pop('download_filenames', None)
        
        if successful_downloads == 0:
            return "No files were found to download", 404
        
        # Return the zip file
        from flask import send_file
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=f'galleryloop_images_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
            mimetype='application/zip'
        )
        
    except Exception as e:
        print(f"Error creating download: {e}")
        return f"Error creating download: {str(e)}", 500

@app.route('/direct-purchase', methods=['POST'])
def direct_purchase():
    if 'email' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    try:
        data = request.get_json()
        user_email = session['email']
        image_id = data.get('image_id')
        filename = data.get('filename')
        price = data.get('price')
        
        # Store single filename in session for immediate download
        if float(price) == 0:
            session['download_filenames'] = [filename]
            return jsonify({
                'success': True, 
                'message': f'Purchase successful! {filename} for ₹{price}',
                'download_required': True
            })
        
        return jsonify({
            'success': True, 
            'message': f'Purchase successful! {filename} for ₹{price}',
            'download_required': False
        })
        
    except Exception as e:
        print(f"Error during direct purchase: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
