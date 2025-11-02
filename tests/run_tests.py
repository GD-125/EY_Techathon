"""
Test Runner - Run all tests
"""
import sys
from pathlib import Path

# Add test directory to path
sys.path.append(str(Path(__file__).parent))

def run_all_tests():
    """Run all test modules"""
    print("\n" + "="*70)
    print("  🧪 Running All Tests")
    print("="*70 + "\n")

    test_results = []

    # Test 1: Credit Scoring Service
    print("📋 Test Suite 1: Credit Scoring Service")
    print("-" * 70)
    try:
        import test_credit_scoring
        print("✅ Credit Scoring Service tests completed\n")
        test_results.append(("Credit Scoring", True))
    except Exception as e:
        print(f"❌ Credit Scoring Service tests failed: {e}\n")
        test_results.append(("Credit Scoring", False))

    # Test 2: Data Processor
    print("\n📋 Test Suite 2: Data Processor")
    print("-" * 70)
    try:
        import test_data_processor
        print("✅ Data Processor tests completed\n")
        test_results.append(("Data Processor", True))
    except Exception as e:
        print(f"❌ Data Processor tests failed: {e}\n")
        test_results.append(("Data Processor", False))

    # Summary
    print("\n" + "="*70)
    print("  📊 Test Summary")
    print("="*70)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name:30} {status}")

    print("-" * 70)
    print(f"  Total: {passed}/{total} test suites passed")
    print("="*70 + "\n")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
