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
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'sen_project'
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
                'profile_img': user1[2] if user1[2] else "static/uploads/default-profile.jpg"
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
            query = "SELECT filename, image_path, price FROM images2 WHERE code = %s LIMIT %s OFFSET %s"
            cur.execute(query, (session['secretCode'], per_page, (page - 1) * per_page))
        else:
            query = "SELECT filename, image_path, price FROM images2 WHERE Status = %s LIMIT %s OFFSET %s"
            cur.execute(query, ('public', per_page, (page - 1) * per_page))

        rows = cur.fetchall()

        if not rows:
            return jsonify({'message': 'No images found'}), 404

        images = []
        for row in rows:
            filename, image_path, price = row  # Ensure correct unpacking
            image_url = f"static/uploads/{filename}"  # Construct the URL for accessing the image
            
            images.append({
                'url': image_url,
                'filename': filename,
                'price': price  # Include price in the response
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
    
    if file and file.filename != '' and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_path = file_path.replace("\\", "/")  # Normalize path for cross-platform support
        file.save(file_path)  # Save the file
        
        relative_path = f"static/uploads/{filename}"  # Ensure it's relative
        
        cur = Mysql.connection.cursor()
        query1 = "SELECT profile_img FROM admin_logins WHERE Email = %s"
        email = session['admin_email']
        cur.execute(query1, (email,))  # Pass the email as a tuple
        row = cur.fetchone()
        
        if row and row[0] and os.path.exists(f"public_html/{row[0]}"):
            os.remove(f"public_html/{row[0]}")
        
        cur.close()
        
        try:
            cur = Mysql.connection.cursor()
            query = "UPDATE admin_logins SET Full_Name = %s, profile_img = %s, bio = %s WHERE Email = %s"
            cur.execute(query, (new_name, relative_path, new_bio, session['admin_email']))
            Mysql.connection.commit()
            
            # Fetch updated profile data
            cur.execute("SELECT Full_Name, bio, profile_img FROM admin_logins WHERE Email = %s", (session['admin_email'],))
            user1 = cur.fetchone()
            
            if user1:
                profile_data = {
                    'name': user1[0], 
                    'bio': user1[1],   
                    'profile_img': user1[2] if user1[2] else "static/uploads/default-profile.jpg"
                }
            else:
                profile_data = None
            
            # Render the adminHome.html page with updated profile data
            return render_template('adminHome.html', profile=profile_data)
        
        except Exception as err:
            return f"Error: {err}", 500
        
        finally:
            cur.close()
    
    return "Invalid file or no file uploaded!", 400
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
            image_url = profile_img if profile_img else "/static/uploads/default-profile.jpg" 
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
    profile_img = profile_img if profile_img else "static/uploads/default-profile.jpg"

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
    data = request.form
    if data:
        print("data received:",data)
        print(session['email'])
    user_email = session['email']
    photographer_email = data.get('photographerEmail')
    event_type = data.get('eventType')
    event_date = data.get('eventDate')
    event_location = data.get('eventLocation')
    contact_info = data.get('contactInfo')
    special_requests = data.get('specialRequests')
    additional_info = data.get('additionalInfo')
    budget = data.get('budget')
    # Insert data into the MySQL database
    cursor = Mysql.connect.cursor()
    query = "INSERT INTO hires (user_email, photographer_email, event_type, event_date, event_location, contact_info, special_requests, additional_info, budget)VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (user_email, photographer_email, event_type, event_date, event_location, contact_info, special_requests, additional_info, budget)
    cursor.execute(query, values)
    cursor.close()
    return jsonify(success=True)

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

if __name__ == '__main__':
    app.run(debug=True)
