import { Component, type ReactNode } from 'react';

interface Props { children: ReactNode }
interface State { error: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error('UI error:', error);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="m-6 p-4 rounded border border-red-300 bg-red-50 text-red-800">
          <div className="font-semibold mb-2">Something went wrong while rendering.</div>
          <pre className="text-xs whitespace-pre-wrap">{this.state.error.message}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}
