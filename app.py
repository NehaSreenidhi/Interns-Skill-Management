from datetime import datetime
import bcrypt
from flask import Flask, flash, redirect, render_template, request, session, url_for, jsonify
import pymongo
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


app = Flask(__name__)
app.secret_key = os.urandom(24)

# Loading SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# MongoDB conncetion
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["myDBase"]
mycol = mydb["interns"]
mentCol = mydb["mentors"]

# FAISS Setup
index = faiss.IndexFlatL2(384) 
model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
interns_metadata = []

index = faiss.IndexFlatL2(384) 
interns_metadata = []  # [(email, skill_name)] mapping

def populate_faiss():
    global interns_metadata
    interns_metadata = []  # Resetting metadata
    index.reset()
    
    interns = mycol.find()
    embeddings = []

    for intern in interns:
        email = intern.get("email", "")
        for skill in intern.get("skills", []):
            skill_name = skill.get("skill_name", "").strip()
            skill_level = skill.get("skill_level", "").strip()
            
            # Creating a more meaningful sentence for embeddings
            skill_description = f"{skill_name} at {skill_level} level"
            
            if skill_description:  # To avoid empty embeddings
                skill_vector = model.encode([skill_description])[0]
                embeddings.append(skill_vector)
                interns_metadata.append((email, skill_name))  # Store email + skill

    if embeddings:
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)  # Adding to FAISS index


@app.route("/")
def home():
    if "user" in session:
        return f"Welcome, {session['user']}!"
    return render_template("landingPage.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        name = request.form["name"]

        if mycol.find_one({"email": email}):
            flash("Email already registered. Please log in.", "warning")
            return redirect("/register")
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        try:
            mycol.insert_one({"email": email, "password": hashed_password, "name":name})
            flash("Registration successful! You can now log in.", "success")
            return redirect("/login")  
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect("/register")
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user_type = request.form.get('userType')

        if user_type == 'Intern':
            user = mycol.find_one({"email": email})
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
                session['email'] = email  
                session['user_type'] = user_type  
                flash('Login successful', 'success')
                return redirect(url_for('intern_dashboard'))  

        elif user_type == 'Mentor':
            mentor = mentCol.find_one({"email": email})
            if mentor and password == mentor['password']:
                session['email'] = email  
                session['user_type'] = 'Mentor'  
                flash('Login successful', 'success')
                return redirect(url_for('mentor_dashboard'))  

        flash('Invalid credentials. Please try again.', 'danger')
        return redirect(url_for('login')) 
    return render_template('login.html')  

@app.route('/reset_password', methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = request.form["email"].strip()
        password = request.form["pwd"].strip()
        confirm_password = request.form["confirm_pwd"].strip()

        user = mycol.find_one({"email": email})
        if not user:
            flash("Email not found!", "danger")
            return redirect(url_for("reset_password"))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for("reset_password"))

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        mycol.update_one({"email": email}, {"$set": {"password": hashed_password}})
        
        flash("Password reset successfully!", "success")
        return redirect(url_for("login")) 

    return render_template("reset_password.html")


@app.route('/intern_dashboard')
def intern_dashboard():
    if "email" not in session:
        return redirect("/login")
    
    user_email = session["email"]
    user = mycol.find_one({"email": user_email})
    if user and "name" in user:
        name = user["name"]
    else:
        name = "Intern"  

    return render_template("intern_dashboard.html", name=name, email = user_email)

@app.route("/add_skills", methods=["POST"])
def add_skills():
    if "email" not in session or session["user_type"] != "Intern":
        return redirect("/login")

    skill_name = request.form["skillName"]
    skill_level = request.form["skillLevel"]
    email = session["email"]

    # Fetching the intern's profile
    intern = mycol.find_one({"email": email})
    if not intern:
        flash("Intern profile not found.", "danger")
        return redirect("/intern_dashboard")

    # Fetching the current skills list
    updated_skills = intern.get("skills", [])

    # Checking for duplicate skill
    if any(skill["skill_name"].lower() == skill_name.lower() for skill in updated_skills):
        flash(f"The skill '{skill_name}' already exists in your profile.", "warning")
        return redirect("/intern_dashboard")
    
    # Creating embedding for the new skill name
    skill_embedding = model.encode([skill_name])[0]  # Encoding - skill name

    # Adding the new skill to the list
    updated_skills.append({"skill_name": skill_name, "skill_level": skill_level})

    # Updating activity log with today's date
    today = datetime.today().strftime("%Y-%m-%d")
    activity_log = intern.get("activity", {})
    activity_log[today] = "add"

    # Updating both the skills and the activity log in db
    try:
        mycol.update_one(
            {"email": email}, 
            {
                "$set": {"skills": updated_skills, "activity": activity_log}
            }, 
            upsert=False
        )

        # Adding the skill's embedding to FAISS index
        index.add(np.array([skill_embedding], dtype=np.float32)) 
        flash(f"Skill '{skill_name}' added successfully!", "success")

    except Exception as e:
        flash(f"Error updating MongoDB: {str(e)}", "danger")
        return redirect("/intern_dashboard")

    return redirect("/intern_dashboard")

@app.route('/get_skills', methods=["GET"])
def get_skills():
    if "email" not in session or session["user_type"] != "Intern":
        return jsonify({"error": "Unauthorized access"}), 401

    email = session["email"]

    # MongoDB Query - retrieve the skills field of the intern
    intern = mycol.find_one({"email": email}, {"_id": 0, "skills": 1})
    if not intern:
        return jsonify({"error": "Intern profile not found"}), 404

    skillsList = intern.get("skills", []) 

    # Data preparation for the pie chart
    pie_chart_data = []
    weight_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}  # Weight for importance

    for skill in skillsList:
        skill_name = skill.get("skill_name", "Unknown Skill")
        skill_level = skill.get("skill_level", "Beginner")  # Default level
        weight = weight_map.get(skill_level, 1)
        pie_chart_data.append({
            "skill_name": skill_name,
            "skill_level": skill_level,
            "weight": weight
        })
    
    # Sort by weight (Advanced > Intermediate > Beginner)
    pie_chart_data = sorted(pie_chart_data, key=lambda x: -x["weight"])
    # Returning the formatted data for the frontend
    return jsonify({
        "skills": pie_chart_data,  # For pie chart
        "skills_list": skillsList  # Raw skill data (without weight)
    }), 200

@app.route('/get_skills/<email>', methods=["GET"])
def get_skills_by_email(email):
    intern = mycol.find_one({"email": email}, {"_id": 0, "skills": 1})
    if not intern:
        return jsonify({"error": "Intern profile not found"}), 404

    skillsList = intern.get("skills", [])
    weight_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}

    pie_chart_data = [{
        "skill_name": skill.get("skill_name", "Unknown Skill"),
        "skill_level": skill.get("skill_level", "Beginner"),
        "weight": weight_map.get(skill.get("skill_level", "Beginner"), 1)
    } for skill in skillsList]

    return jsonify({
        "skills": sorted(pie_chart_data, key=lambda x: -x["weight"])
    }), 200


@app.route('/modify_skills', methods=['POST'])
def modify_skills():
    if "email" not in session or session["user_type"] != "Intern":
        return redirect("/login")

    email = session["email"]
    existing_skill_name = request.form.get('existingSkill')
    new_skill_level = request.form.get('newSkillLevel')

    # Updating the skill level inside the intern's skills list
    update_result = mycol.update_one(
        {"email": email, "skills.skill_name": existing_skill_name},  
        {"$set": {"skills.$.skill_level": new_skill_level}}  
    )

    if update_result.modified_count > 0:
        # Log modification activity
        today = datetime.today().strftime("%Y-%m-%d")
        
        # Fetching the intern's profile to update activity
        intern = mycol.find_one({"email": email})
        
        if intern:
            activity_log = intern.get("activity", {})
            activity_log[today] = "modify"
            
            # Update activity log in the database
            mycol.update_one({"email": email}, {"$set": {"activity": activity_log}})

            flash(f"Skill '{existing_skill_name}' updated successfully!", "success")
        else:
            flash("Intern profile not found.", "danger")
    else:
        flash("Skill update failed. Ensure the skill exists.", "danger")

    return redirect(url_for('intern_dashboard'))


@app.route('/mentor_dashboard')
def mentor_dashboard():
    if "email" not in session: 
        return redirect("/login")
    user_email = session["email"]
    user = mentCol.find_one({"email": user_email})
    if user and "name" in user:
        name = user["name"]
    else:
        name = "Mentor" 
    ints = mycol.find()
    interns_list = list(ints)
    return render_template('mentor_dashboard.html', interns_list=interns_list, name = name)


@app.route('/profile/<email>')
def intern_profile(email):
    if "email" not in session:
        return redirect("/login")
    
    user_email = session["email"]
    user_type = session.get("user_type")

    intern = mycol.find_one({"email": email}, {"_id": 0, "name": 1, "email": 1, "activity": 1})
    curr_intern = mycol.find_one({"email": email})

    if not intern:
        return "Profile not found", 404
    
    activity_data = intern.get("activity", {}) 

    # Converting activity data for the heatmap
    heatmap = {}
    for date_str, action in activity_data.items():
        heatmap[date_str] = "high" if action in ["add", "modify"] else "low"

    if user_type == "Intern":
        return render_template('intern_profile.html', intern=intern, email = user_email, heatmap=heatmap, curr_intern=curr_intern, user_type=user_type)
    else:
        return render_template('intern_profile.html', intern=intern, email = email, heatmap=heatmap, curr_intern=curr_intern, user_type=user_type)


@app.route('/search_interns', methods=["GET", "POST"])
def search_interns():
    populate_faiss()
    if "email" not in session:  
        return redirect("/login")

    user_email = session["email"]
    user = mentCol.find_one({"email": user_email})
    name = user["name"] if user and "name" in user else "Mentor"
    
    search_results = []
    # DISTANCE_THRESHOLD = 0.3  # Set a threshold for similarity

    if request.method == "POST":
        search_query = request.form.get("searchQuery", "").strip()

        if search_query:
            if index.ntotal == 0:
                flash("No intern skills found. Please add skills before searching.", "warning")
                return render_template("search_interns.html", name=name, search_results=[])

            # Encode query for search
            query_vector = np.array(model.encode([f"Find interns skilled in {search_query}"])).astype('float32')

            # Perform FAISS search
            D, I = index.search(query_vector, k=min(3, index.ntotal))
            seen_emails = set()  # Prevent duplicates
            for idx, distance in zip(I[0], D[0]):  # Iteration over results
                if 0 <= idx < len(interns_metadata):  # Ensure it's a valid & relevant match
                    intern_email, skill_name = interns_metadata[idx]  # Get email + skill
                    
                    if intern_email not in seen_emails:  # Avoiding duplicates
                        intern = mycol.find_one({"email": intern_email})
                        if intern:
                            search_results.append({
                                "name": intern.get("name", "Unknown"),
                                "email": intern_email,
                                "skills": intern.get("skills", []),
                                "matched_skill": skill_name  
                            })
                            seen_emails.add(intern_email)  # Mark as added
                else:
                    print(f"Filtered out: Index {idx}, Distance {distance}")
    return render_template("search_interns.html", name=name, search_results=search_results)


@app.route("/profile_settings", methods=["GET", "POST"])
def profile_settings():
    if "email" not in session:
        return redirect("/login")
    
    user_email = session["email"]
    user = mycol.find_one({"email": user_email})
    if user and "name" in user:
        name = user["name"]
    else:
        name = "Intern"  

    if request.method == "POST":
        phone = request.form["phone"]
        linkedin = request.form["linkedin"]
        github = request.form["github"]
        leetcode = request.form["leetcode"]

        user_details = {
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
            "leetcode": leetcode
        }
        mycol.update_one({"email": user_email}, {"$set": user_details}, upsert=True)
        return redirect("/intern_dashboard")
    return render_template("profile_settings.html", name=name, email = user_email)

@app.route("/logout")
def logout():
    session.clear() 
    flash("You have been logged out.", "info")
    return redirect("/login") 

if __name__ == "__main__":
    app.run(debug=True)
