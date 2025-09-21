# script/setup_db.py
from db_manager import init_db, add_user

if __name__ == "__main__":
    init_db()
    ok = add_user("admin", "admin123", role="admin")
    if ok:
        print("Database created and default admin added (admin/admin123).")
    else:
        print("Database created. Admin user already exists.")
