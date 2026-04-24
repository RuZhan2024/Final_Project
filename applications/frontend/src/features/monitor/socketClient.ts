import { buildMonitorWebSocketUrl } from "./api";

/**
 * WebSocket transport wrapper for live monitor predictions.
 *
 * The live monitor UI needs one request/one response semantics over a persistent
 * socket. This client enforces that contract and retries only transient socket
 * failures.
 */
function makeAbortError(message: string) {
  /** Tag logical mode-switch shutdowns so callers do not retry them as faults. */
  const err = new Error(message);
  err.name = "AbortError";
  return err;
}

export function createMonitorSocketClient({
  apiBase,
  connectTimeoutMs,
  predictTimeoutMs,
  WebSocketImpl = WebSocket,
}: {
  apiBase: string;
  connectTimeoutMs: number;
  predictTimeoutMs: number;
  WebSocketImpl?: typeof WebSocket;
}) {
  /**
   * Create a single-flight prediction client over the monitor WebSocket route.
   *
   * At most one request may be pending at a time. Callers that need concurrent
   * prediction must open separate clients instead of multiplexing this one.
   */
  let socket: WebSocket | null = null;
  let ready = false;
  let closeState = { message: "WebSocket closed", abort: false };
  let pending: {
    resolve?: (message: any) => void;
    reject?: (error: any) => void;
  } | null = null;

  const clearSocket = () => {
    // Keep readiness derived from the current socket instance so reconnects do
    // not accidentally reuse OPEN state from a previous connection.
    socket = null;
    ready = false;
  };

  const rejectPending = (error: unknown) => {
    if (!pending) return;
    const current = pending;
    pending = null;
    current.reject?.(error);
  };

  const close = (reason = "WebSocket closed during mode switch") => {
    // Close is also used as a logical abort when monitor mode changes, so the
    // pending request is rejected with AbortError semantics rather than a retry.
    closeState = { message: reason, abort: true };
    rejectPending(makeAbortError(reason));
    if (socket) {
      try {
        socket.close();
      } catch {
        // ignore
      }
    }
    clearSocket();
  };

  const ensureOpen = async () => {
    /** Open the socket if needed and resolve only once the connection is ready. */
    if (socket && ready && socket.readyState === WebSocketImpl.OPEN) {
      return socket;
    }

    const ws = new WebSocketImpl(buildMonitorWebSocketUrl(apiBase));
    socket = ws;
    ready = false;

    return await new Promise<WebSocket>((resolve, reject) => {
      let settled = false;
      const timeoutId = window.setTimeout(() => {
        if (settled) return;
        settled = true;
        try {
          ws.close();
        } catch {
          // ignore
        }
        clearSocket();
        reject(new Error("WebSocket connect timeout"));
      }, connectTimeoutMs);

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeoutId);
        ready = true;
        resolve(ws);
      };

      ws.onerror = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeoutId);
        clearSocket();
        reject(new Error("WebSocket connect failed"));
      };

      ws.onclose = () => {
        // A close event must reject the pending request; otherwise the caller
        // would wait forever for a response that can no longer arrive.
        const error = closeState.abort ? makeAbortError(closeState.message) : new Error(closeState.message);
        closeState = { message: "WebSocket closed", abort: false };
        clearSocket();
        rejectPending(error);
      };

      ws.onmessage = (event) => {
        if (!pending) return;
        try {
          const data = JSON.parse(String(event?.data || "{}"));
          if (data?.error) {
            pending.reject?.(new Error(String(data?.detail || "predict_window ws error")));
          } else {
            pending.resolve?.(data);
          }
        } catch (err) {
          pending.reject?.(err);
        } finally {
          pending = null;
        }
      };
    });
  };

  const sendOnce = async (payload: unknown) => {
    /** Send one prediction payload and resolve on the next matching response. */
    const ws = await ensureOpen();
    return await new Promise<any>((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        pending = null;
        try {
          ws.close();
        } catch {
          // ignore
        }
        clearSocket();
        reject(new Error("WebSocket predict timeout"));
      }, predictTimeoutMs);

      pending = {
        resolve: (message) => {
          window.clearTimeout(timeoutId);
          resolve(message);
        },
        reject: (error) => {
          window.clearTimeout(timeoutId);
          reject(error);
        },
      };

      try {
        // `pending` is assigned before send so a same-tick reply cannot arrive
        // before the client has somewhere to resolve it.
        ws.send(JSON.stringify(payload));
      } catch (err) {
        pending = null;
        window.clearTimeout(timeoutId);
        reject(err);
      }
    });
  };

  const isTransientSocketError = (error: unknown) => {
    const message = String((error as any)?.message || error || "");
    return /WebSocket predict timeout|WebSocket closed|WebSocket connect failed/i.test(message);
  };

  const predict = async (payload: unknown) => {
    /** Retry once on transient socket failures, but not on logical aborts. */
    try {
      return await sendOnce(payload);
    } catch (err) {
      if (!isTransientSocketError(err)) throw err;
      close("WebSocket closed");
      return await sendOnce(payload);
    }
  };

  return {
    close,
    predict,
  };
}
