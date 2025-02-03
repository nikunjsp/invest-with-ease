import smtplib
import pandas as pd
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart


# def email_alert(subject, body, to):
msg = MIMEMultipart()
msg.set_content(body)
msg['subject'] = "Daily Actions"
msg['to'] = "nikunjsp2212@gmail.com"
df_test = pd.read_csv('op.csv')
user = "investwithease11@gmail.com"
msg['from'] = user
password = "wuooedwlfdgzjsnf"
html = """\
<html>
<head></head>
<body>
    {0}
</body>
</html>
""".format(df_test.to_html())

part1 = MIMEText(html, 'html')
msg.attach(part1)

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(user, password)
# server.send_message(msg.as_string())
server.sendmail(msg['From'], msg['to'], msg.as_string())
server.quit()

'''
if __name__ == '__main__':
    email_alert("Daily Actions", df_test, "nikunjsp2212@gmail.com")'''
