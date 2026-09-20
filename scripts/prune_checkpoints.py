import argparse
from _common import run, load_settings, create_pool, Database, create_checkpointer
from src.checkpoint_retention import CheckpointRetention


def main():
    parser = argparse.ArgumentParser(description='Preview by default; never invokes an LLM. Pauses while any graph invocation is active.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--apply', action='store_true')
    parser.add_argument('--days', type=int)
    parser.add_argument('--force', action='store_true', help='Explicitly allow deletion without a saved episodic summary.')
    parser.add_argument('--current-thread', help='Streamlit thread ID to protect, even when inactive.')
    parser.add_argument('--user-id', help='Restrict cleanup to one local user identity.')
    args = parser.parse_args()
    settings = load_settings()
    with create_pool(settings) as pool:
        retention = CheckpointRetention(Database(pool), create_checkpointer(pool), settings)
        preview = retention.prune(args.current_thread, args.user_id, args.days, args.force)
        if args.current_thread is None:
            # Without a browser thread ID, conservatively protect ALL active conversations.
            active = [row for row in preview['candidates'] if row['status'] == 'active']
            preview['skipped'].extend({'thread_id': row['thread_id'], 'reason': 'active conversation; no --current-thread supplied'} for row in active)
            preview['candidates'] = [row for row in preview['candidates'] if row['status'] == 'archived']
        for row in preview['candidates']:
            print(f"CANDIDATE {row['thread_id']} last_activity={row['last_activity'].isoformat()} checkpoints≈{row['checkpoint_count']} reason={row['reason']}")
        for row in preview['skipped']:
            print(f"SKIP {row['thread_id']}: {row['reason']}")
        if args.apply:
            result = retention.prune(args.current_thread, args.user_id, args.days, args.force, apply=True,
                                     only_ids={row['thread_id'] for row in preview['candidates']})
            print('Deleted:', result['deleted'], 'Failed:', result['failed'])
            return 1 if result['failed'] else 0
        print('Dry run only. Use --apply --current-thread THREAD_ID after reviewing candidates.')
    return 0


if __name__ == '__main__':
    raise SystemExit(run(main))
