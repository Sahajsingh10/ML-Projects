#author Sahaj Singh
def add():
    file_read = open('address book', 'r')
    lines = file_read.readlines()
    with open('address book', 'w') as outFile:
        if len(lines) > 0:
            outFile.writelines(lines)
        else:
            outFile.write('Name')
            outFile.write(" ")
            outFile.write('Email')
            outFile.write(" ")
            outFile.write('Phone')
            outFile.write(" ")
        while True:
            s = input('Do you want to add? ')
            if s == 'yes' or s == 'Yes' or s == 'y' or s == 'YES':
                outFile.write('\n')
                name = input('enter the name of the contact')
                while name == " " or name == "":
                    name = input('enter the name of the contact')
                email = input('enter the email of the contact')
                while email == " " or email == "":
                    email = input('enter the email of the contact')
                num = input('enter the phone number of the contact starting with +1 ')
                while True:
                    if num == " " or num == "" or num.startswith('+1') is False or len(num) > 12:
                        num = input('Enter a valid phone number, enter the phone number of the contact')
                    else:
                        break
                outFile.write(name)
                outFile.write(" ")
                outFile.write(email)
                outFile.write(" ")
                outFile.write(num)
                outFile.write(" ")
            elif ans == 'no' or 'No' or 'n' or 'NO':
                break


def search():
    with open('address book', 'r') as outFile:
        data = outFile.read()

        print('How do you want to search')
        print("1. By Name")
        print("2. By email")
        print("3. By number")
        op = input("Enter a number between 1,2 and 3")
        if op == '1':
            n = input("enter the name of the person")
            if n.upper() or n.lower() or n.capiltalize() or n in data:
                print("yes found it")
            else:
                print("not found")
        if op == '2':
            e = input("enter the email of the person")
            if e.upper() or e.lower() or e.capiltalize() or e in data:
                print("yes found it")
            else:
                print("not Found")
        if op == '3':
            num = input("enter the email of the person")
            if e in data:
                print("yes found it")
            else:
                print("not Found")


def write_update(lines: list):
    with open('address book', 'w') as outFile:
        outFile.writelines(lines)


def update():
    with open('address book', 'r') as outFile:
        lines = outFile.readlines()
        print('How do you want to update')
        print("1. By Name")
        print("2. By email")
        print("3. By number")
        op = input("Enter a number between 1,2 and 3")
        if op == '1':
            n = input("enter the name you want to update")
            r = input('enter the replacement')
            found = False
            for i, line in enumerate(lines):
                splits = line.split(" ")
                name = splits[0]
                if name == n:
                    splits[0] = r
                    lines[i] = ' '.join(splits)
                    found = True
                    break

            if found:
                write_update(lines)
            else:
                print('Could not find that name')
        if op == '2':
            n = input("enter the email you want to update")
            r = input('enter the replacement')
            found = False
            for i, line in enumerate(lines):
                splits = line.split(" ")
                email = splits[1]
                if email == n:
                    splits[1] = r
                    lines[i] = ' '.join(splits)
                    found = True
                    break

            if found:
                write_update(lines)
            else:
                print('Could not find that email')
        if op == '3':
            n = input("enter the phone you want to update")
            r = input('enter the replacement')
            found = False
            for i, line in enumerate(lines):
                splits = line.split(" ")
                phone = splits[2]
                if phone == n:
                    splits[2] = r
                    lines[i] = ' '.join(splits)
                    found = True
                    break

            if found:
                write_update(lines)
            else:
                print('Could not find that phone')


def delete():
    with open('address book', 'r') as outFile:
        lines = outFile.readlines()

        print('How do you want to delete? ')
        print("1. By Name")
        print("2. By email")
        print("3. By number")
        op = input("Enter a number between 1,2 and 3")

        if op == '1':
            n = input("enter the name of the person ")
            for i, line in enumerate(lines):
                name = line.split(' ')[0]
                found = False
                if name == n:
                    lines.pop(i)
                    found = True
            if found:
                write_update(lines)
            else:
                print('Cannot find someone with that name')
        elif op == '2':
            n = input("enter the email of the person")
            for i, line in enumerate(lines):
                email = line.split(' ')[1]
                found = False
                if email == n:
                    lines.pop(i)
                    found = True
            if found:
                write_update(lines)
            else:
                print('Cannot find someone with that email')
        elif op == '3':
            n = input("enter the phone of the person")
            for i, line in enumerate(lines):
                phone = line.split(' ')[2]
                found = False
                if phone == n:
                    lines.pop(i)
                    found = True
            if found:
                write_update(lines)
            else:
                print('Cannot find someone with that phone')


def display():
    with open('address book', 'r') as outFile:
        data = outFile.read()
        for i in data:
            print(i, end="")
        print()


while True:
    ans = input('Do you want to start? ')
    if ans == 'yes' or ans == 'Yes' or ans == 'y' or ans == 'YES':
        ques = input(
            'what do you want to do:\n 1. Write in the file\n 2. search the file\n 3. Delete an entry\n 4. Display the file\n 5. update entry')
        if ques == '1':
            add()
        elif ques == '2':
            search()
        elif ques == '3':
            delete()
        elif ques == '4':
            display()
        elif ques == '5':
            update()
    elif ans == 'no' or 'No' or 'n' or 'NO':
        break
