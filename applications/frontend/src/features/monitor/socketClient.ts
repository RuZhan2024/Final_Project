import { buildMonitorWebSocketUrl } from "./api";

function makeAbortError(message: string) {
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
  let socket: WebSocket | null = null;
  let ready = false;
  let closeState = { message: "WebSocket closed", abort: false };
  let pending: {
    resolve?: (message: any) => void;
    reject?: (error: any) => void;
  } | null = null;

  const clearSocket = () => {
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
