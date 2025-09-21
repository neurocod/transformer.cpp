#pragma once

class BinaryWriter {
private:
		std::ofstream& _stream;
		
		void writeBytes(std::span<const std::byte> bytes) {
			_stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
		}
public:
		explicit BinaryWriter(std::ofstream& stream) : _stream(stream) {}
		
		// Write any trivially copyable type
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void write(const T& value) {
				writeBytes(std::as_bytes(std::span(&value, 1)));
		}
		
		// Write span of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void writeSpan(std::span<const T> data) {
				writeBytes(std::as_bytes(data));
		}
		
		// Write vector of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void writeVector(const std::vector<T>& data) {
				writeSpan(std::span(data));
		}

		void write(const std::string& str) {
			const uint32_t length = static_cast<uint32_t>(str.size());
			write(length);
			if (length > 0) {
				_stream.write(str.data(), length);
			}
		}
			
		// Check stream state
		bool ok() const { return _stream.good(); }
};