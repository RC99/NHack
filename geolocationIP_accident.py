import requests
import smtplib
from email.mime.text import MIMEText

# Get the public IP address
public_ip_response = requests.get('https://api.ipify.org?format=json')
public_ip = public_ip_response.json()['ip']

# Use a geolocation API to get the location of the public IP
location_response = requests.get(f'http://ip-api.com/json/{public_ip}')
location_data = location_response.json()

if location_data['status'] == 'success':
    lat = location_data['lat']
    long = location_data['lon']
    
    print(f"Latitude: {lat}, Longitude: {long}")

    # Check latitude and longitude conditions
    if 30 < lat < 35 and -119 < long < -118:
        # Send a normal email notification
        sender_email = "email_your@gmail.com"
        receiver_email = "email_to@usc.edu"  # Replace with your recipient's email
        password = "app password"  # Use your email password or app password

        # Create the email content for normal email
        email_msg = MIMEText("Alert: Your specified condition has been met.")
        email_msg["Subject"] = "Location Alert"
        email_msg["From"] = sender_email
        email_msg["To"] = receiver_email

        # Send the email
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.sendmail(sender_email, receiver_email, email_msg.as_string())
            print("Email notification sent!")
        except Exception as e:
            print(f"Error sending email: {e}")

        # Send an SMS via email
        sms_receiver_number = "your_number@msg.fi.google.com"  # Replace with the recipient's Google Fi number
        
        # Create the SMS content
        sms_msg = MIMEText("This is a normal SMS alert.")
        sms_msg["Subject"] = "SMS Alert"
        sms_msg["From"] = sender_email
        sms_msg["To"] = sms_receiver_number

        # Send the SMS
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, password)
                server.sendmail(sender_email, sms_receiver_number, sms_msg.as_string())
            print("SMS sent successfully!")
        except Exception as e:
            print(f"Error sending SMS: {e}")

else:
    print("Could not get location data.")
