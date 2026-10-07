"""Run all plotting scripts (Q4a-d, Q5)."""

from plot_q4 import plot_q4a
from plot_q4b import plot_q4b
from plot_q4c import plot_q4c
from plot_q4d import plot_q4d
from plot_q5 import plot_q5


def main():
    scripts = [
        ("Q4a", plot_q4a),
        ("Q4b", plot_q4b),
        ("Q4c", plot_q4c),
        ("Q4d", plot_q4d),
        ("Q5", plot_q5),
    ]

    for name, fn in scripts:
        print(f"\n{'='*50}")
        print(f"  Generating {name} plots")
        print(f"{'='*50}")
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {name} failed: {e}")

    print("\nAll plots done. Check results/plots/")


if __name__ == "__main__":
    main()
