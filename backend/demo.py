"""
Demo Script - Test the Loan ERP System
Run this to see the system in action without frontend
"""

import requests
import time
import json
from datetime import datetime


BASE_URL = "http://localhost:8000"


def print_separator():
    print("\n" + "="*70 + "\n")


def print_message(role, message):
    if role == "user":
        print(f"👤 Customer: {message}")
    else:
        print(f"🤖 Assistant:\n{message}")
    print()


def test_health():
    """Test if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy and running!")
            return True
        else:
            print("❌ Server responded but with error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("Please ensure the backend is running: python main.py")
        return False


def demo_scenario(phone, amount, tenure, customer_name):
    """Run a complete demo scenario"""
    print_separator()
    print(f"🎬 DEMO SCENARIO: {customer_name}")
    print_separator()

    # Start session
    print("📞 Starting chat session...")
    response = requests.post(f"{BASE_URL}/api/chat/start")
    data = response.json()
    session_id = data['session_id']
    print(f"✅ Session ID: {session_id}\n")

    print_message("assistant", data['message'])

    # Send messages
    messages = [
        (phone, 2.0),
        (amount, 2.5),
        (tenure, 2.0),
        ("yes", 1.5)
    ]

    for message, response_time in messages:
        time.sleep(0.5)  # Simulate thinking time

        print_message("user", message)

        response = requests.post(
            f"{BASE_URL}/api/chat/message",
            json={
                "session_id": session_id,
                "message": message,
                "response_time": response_time
            }
        )

        data = response.json()
        print_message("assistant", data['message'])

        if data.get('personality_detected'):
            print(f"🎭 Personality Detected: {data['personality_detected'].upper()}")
            print()

        if data.get('application_id'):
            print(f"📋 Application ID: {data['application_id']}")
            print()

    # Get session summary
    print_separator()
    print("📊 SESSION SUMMARY")
    print_separator()

    response = requests.get(f"{BASE_URL}/api/chat/session/{session_id}/summary")
    summary = response.json()

    if summary['success']:
        data = summary['data']
        print(f"Session ID: {data['session_id']}")
        print(f"Application ID: {data.get('application_id', 'N/A')}")
        print(f"Status: {data['state']}")
        print(f"Personality Type: {data['personality_type']}")
        print(f"Behavioral Score: {data['behavioral_score']:.1f}/100")
        print(f"Total Messages: {data['total_messages']}")

    print_separator()


def show_statistics():
    """Show system statistics"""
    print_separator()
    print("📈 SYSTEM STATISTICS")
    print_separator()

    response = requests.get(f"{BASE_URL}/api/stats")
    stats = response.json()

    if stats['success']:
        data = stats['data']
        print(f"Total Applications: {data['total_applications']}")
        print(f"Approved: {data['approved']}")
        print(f"Rejected: {data['rejected']}")
        print(f"Pending: {data['pending']}")
        print(f"Approval Rate: {data['approval_rate']:.1f}%")

    print_separator()


def main():
    print("\n" + "="*70)
    print("  🏦 LOAN ERP SYSTEM - AUTOMATED DEMO")
    print("  Tata Capital - AI Loan Assistant")
    print("="*70 + "\n")

    # Check server health
    if not test_health():
        return

    print("\n🎯 Running Demo Scenarios...\n")

    # Scenario 1: Excellent Credit (Will be approved)
    demo_scenario(
        phone="9876543210",
        amount="500000",
        tenure="36",
        customer_name="Raj Kumar (Credit Score: 780 - Excellent)"
    )

    input("\n⏸️  Press Enter to continue to next scenario...\n")

    # Scenario 2: Good Credit (Will be approved)
    demo_scenario(
        phone="9123456789",
        amount="800000",
        tenure="48",
        customer_name="Priya Sharma (Credit Score: 720 - Good)"
    )

    input("\n⏸️  Press Enter to continue to next scenario...\n")

    # Scenario 3: Low Credit (May be rejected)
    demo_scenario(
        phone="9988776655",
        amount="600000",
        tenure="36",
        customer_name="Amit Patel (Credit Score: 620 - Low)"
    )

    # Show final statistics
    show_statistics()

    print("\n✅ Demo completed successfully!")
    print("\n💡 TIP: Open http://localhost:8000/docs to explore the API\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        print("Please check if the backend server is running!")
