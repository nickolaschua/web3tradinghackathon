# Hackathon Guide: How to Sign In AWS and Launch Your Bot

FOLLOW THIS PROCESS TO ACCESS YOUR CLOUD.

## 1. Sign in to the AWS Portal

1. **Open the Link:** Use the portal URL provided :  https://d-906625dad1.awsapps.com/start
2. Follow the steps using your given email as username and verify.
3. Set a new password.
4. Enter your username/email and password.
5. Set up the MFA using any method of your choice.
6. You’ll land on the **AWS Access Portal**.
    - Click the **AWS Account** tile.
    - On the next line (e.g., Hackathon-TeamX), click the Management Console link (HackathonPermissionSet) to open the AWS console.

---

## 2. Navigate to EC2 in `ap-southeast-2`

This is a critical first step. Our entire event runs in the Sydney **(**`ap-southeast-2`**)** region.

1. In the top-right corner of the AWS Console, check that the region is set to `ap-southeast-2`. If it says something else (like "Ohio"), click it and change it to `ap-southeast-2`.
2. In the main search bar, type **EC2** and select it from the results to open the EC2 Dashboard.

---

## 3. Launch an Instance Using the Template

1. In the left menu, click **Instances → Launch Instance (Dropdown Menu)**.
2. Select **Launch from template**.
3. Pick the pre configured template (e.g. `HackathonBotTemplate`).
    - This has the correct AMI, instance type (e.g. `t3.medium`), storage, and networking pre-set.
4. **Name:** Give your instance a name (e.g., `my-trading-bot`).
5. **Key pair (login):**
    - Click the "Key pair" dropdown
    - Proceed without a key pair : *it is not necessary.*
    - We are using a secure browser based connection, so an SSH key is not needed.
6. Leave other things as default
7. Click **Launch Instance**.

---

## 4. Connect to Your Instance (Session Manager Only)

Your *only* method for connecting is Session Manager. 

SSH and EC2 Instance Connect are blocked for security.

1. After launching, click **View all instances**.
2. Wait 1-2 minutes for your instance's "Status check" to show "2/2 or 3/3 checks passed".
3. Select your instance by clicking the checkbox next to its name.
4. At the top of the page, click the **Connect** button.
5. Select the **Session Manager** tab.
6. Click the orange **Connect** button.

A new browser tab will open with a black terminal screen. You are now securely connected to your instance.

---

## 5. Setup Your Bot

1. Once inside the instance terminal, you have `sudo` (root) access.
2. type this and enter

```bash
cd ~

```

this takes you to ***your home directory***.

you can check your present working directory via : *pwd*.

### Clone your code from GitHub, etc.:Bash

```bash
git clone <your-bot-repo>
cd <repo>
pip install -r requirements.txt
```

- Install any additional dependencies your bot needs, We have preinstalled somethings like git , python via launch template.
- Configure your bot (API keys, trading parameters, etc.) using environment variables or a config file.

## 6. How to run it.

1. Once you pull your code and you are ready to launch it for trading live. 
2. If you run your agent without a remote host (tmux) , it would stop when you exit AWS.
3. use TMUX to run the agent/bot on **Remote host** 
4. Use these commands.

```bash
# Update packages
sudo dnf update -y

# Install tmux
sudo dnf install -y tmux

# Verify installation
tmux -V
```

```bash
tmux
```

A remote session is opened.

Now,

to detach out of this session.

```bash
ctrl + B , then press D
```

- **`detach`** = exit tmux without killing the session running in the background
- **`exit` inside tmux** = ends the session completely.

if you leave without detaching , the process will be halted.

To reattach :

```bash
# if you have only one session running
tmux attach 

# to attach one more process in the background 
tmux attach -t <session-name>
# or
tmux attach-session -t <session-name>

# session names are generally 0,1,2 and so on
```

---

## 7. Managing Your Instance

- **Start/Stop**: You can stop the instance when not in use (saves cost).
- **Terminate**: Deletes the instance permanently. Don’t terminate unless done with the hackathon.
- **Tags**: Tag your instance with your team name for easier identification.

---

## 8. The Rules & Limits (Please Read!)

To ensure fairness and security, your account has strict limits. Please don't waste time trying to work around them.

- **Region:** You can **only** use the **`ap-southeast-2` (Sydney)** region.
- **Instance Type:** You can **only** launch **`t3.medium`** instances.
- **Connection:** You can **only** connect via **Session Manager**.
- **Storage:** Your instance's disk (EBS Volume) is limited to 3**0 GB**.
- **Other Services:** You cannot use other AWS services (like IAM, S3, etc.). Your permissions are restricted to launching and managing your EC2 instance.(just for deploying your bots)
- **Instances : Y**ou should launch only **one** instance inside your EC2 , creating more than 1 will automatically trigger termination since you have access to launch only one instance.