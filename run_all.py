"""Run inference agent on all difficulties and write results to a file."""
import sys
import io


def main() -> None:
    """Execute inference on all difficulty levels and save output."""
    # Redirect stdout to capture all output
    output = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = output

    from inference import run_local  # noqa: E402

    for diff in ["easy", "medium", "hard"]:
        print(f"\n{'#'*60}")
        print(f"# {diff.upper()} DIFFICULTY")
        print(f"{'#'*60}")
        run_local(diff)

    sys.stdout = old_stdout
    result = output.getvalue()

    # Write to file
    with open("results_all.txt", "w", encoding="utf-8") as f:
        f.write(result)

    # Also print to console
    print(result)
    print("\n[Results saved to results_all.txt]")


if __name__ == "__main__":
    main()
